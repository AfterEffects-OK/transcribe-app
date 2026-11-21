import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { GoogleGenAI } from '@google/genai';

const execFileAsync = promisify(execFile);
const app = express();
const port = Number(process.env.PORT || 3000);

if (!process.env.GEMINI_API_KEY) {
  console.error('環境変数 GEMINI_API_KEY が設定されていません。');
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const uploadDir = path.join(os.tmpdir(), 'gemini-transcribe-uploads');
fs.mkdirSync(uploadDir, { recursive: true });

const upload = multer({
  dest: uploadDir,
  limits: {
    // 2時間超の動画/音声も扱えるよう上限を大きめに設定（必要に応じて調整）
    fileSize: 4 * 1024 * 1024 * 1024,
  },
});

app.use(express.static(path.join(process.cwd(), 'public')));

// ffmpeg が利用可能か確認
async function hasFfmpeg() {
  try {
    await execFileAsync('ffmpeg', ['-version']);
    return true;
  } catch {
    return false;
  }
}

// 音声/動画を30分（1800秒）単位のMP3チャンクに分割（ffmpeg利用時）
async function splitMediaIntoChunks(inputPath, chunkSeconds = 1800) {
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chunks-'));
  const outputPattern = path.join(workDir, 'part_%03d.mp3');

  await execFileAsync('ffmpeg', [
    '-i',
    inputPath,
    '-vn',
    '-ac',
    '1',
    '-ar',
    '16000',
    '-b:a',
    '64k',
    '-f',
    'segment',
    '-segment_time',
    String(chunkSeconds),
    '-reset_timestamps',
    '1',
    outputPattern,
  ]);

  const files = fs
    .readdirSync(workDir)
    .filter((f) => f.startsWith('part_') && f.endsWith('.mp3'))
    .sort()
    .map((f) => path.join(workDir, f));

  return { workDir, chunkPaths: files };
}

// Files API アップロード後に ACTIVE 状態になるまで待機
async function uploadAndWaitActive(filePath, mimeType) {
  const uploaded = await ai.files.upload({
    file: filePath,
    mimeType,
  });

  let current = uploaded;
  while (current.state === 'PROCESSING') {
    await new Promise((r) => setTimeout(r, 3000));
    current = await ai.files.get({ name: current.name });
  }

  if (current.state !== 'ACTIVE') {
    throw new Error(`ファイル処理に失敗しました (state: ${current.state})`);
  }

  return current;
}

function buildPrompt({ glossary, partIndex, totalParts, chunkSeconds }) {
  const startSec = (partIndex - 1) * chunkSeconds;
  const startMin = Math.floor(startSec / 60);
  const partInfo =
    totalParts > 1
      ? `これは全体の第${partIndex}/${totalParts}パート（おおよそ${startMin}分付近からの区間）です。`
      : '全体の音声です。';

  const glossaryBlock = glossary?.trim()
    ? `\n【専門用語・固有名詞・話者メモ】\n${glossary.trim()}\n上記を参考に、表記ゆれや聞き間違いを補正してください。`
    : '';

  return `${partInfo}
以下の講義・会議の音声を文字起こししてください。

【厳守ルール】
- 要約、言い換え、省略、中略（「〜など」「以下略」など）は一切せず、発言内容をできる限り忠実に全文起こししてください。
- 話題の区切りや重要な発言ごとに [MM:SS] 形式でタイムスタンプを付けてください（パート分割時はそのパート先頭からの経過時間で構いません）。
- 複数人が話す場合は文脈から「話者A」「話者B」のように区別してください。
- フィラー（「えー」「あのー」など）は読みやすさを損なわない範囲で自然に整えても構いませんが、意味のある発言は削らないでください。
- 聞き取れない箇所は [聞き取り不明] と明記し、推測で捏造しないでください。${glossaryBlock}`;
}

async function transcribeSingleFile(filePath, mimeType, promptText) {
  const uploaded = await uploadAndWaitActive(filePath, mimeType);
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [uploaded, promptText],
    });
    return response.text || '';
  } finally {
    try {
      await ai.files.delete({ name: uploaded.name });
    } catch {
      // 削除失敗は無視
    }
  }
}

function safeUnlink(targetPath) {
  if (!targetPath) return;
  try {
    fs.rmSync(targetPath, { recursive: true, force: true });
  } catch {
    // 無視
  }
}

app.post('/api/transcribe', upload.single('media'), async (req, res) => {
  const inputPath = req.file?.path;
  const originalMime = req.file?.mimetype || 'audio/mpeg';
  const glossary = typeof req.body?.glossary === 'string' ? req.body.glossary : '';
  const useChunking = req.body?.useChunking === 'true';
  const chunkSeconds = 1800; // 30分

  if (!inputPath) {
    return res.status(400).json({ error: '音声または動画ファイルが選択されていません。' });
  }

  let chunkWorkDir = null;

  try {
    const ffmpegAvailable = useChunking ? await hasFfmpeg() : false;

    if (useChunking && ffmpegAvailable) {
      const { workDir, chunkPaths } = await splitMediaIntoChunks(inputPath, chunkSeconds);
      chunkWorkDir = workDir;

      if (chunkPaths.length > 0) {
        const partsText = [];
        for (let i = 0; i < chunkPaths.length; i++) {
          const prompt = buildPrompt({
            glossary,
            partIndex: i + 1,
            totalParts: chunkPaths.length,
            chunkSeconds,
          });
          const text = await transcribeSingleFile(chunkPaths[i], 'audio/mp3', prompt);
          partsText.push(`=== パート ${i + 1} / ${chunkPaths.length} ===\n\n${text.trim()}`);
        }
        return res.json({
          transcript: partsText.join('\n\n---\n\n'),
          mode: `30分分割処理（全${chunkPaths.length}パート）`,
        });
      }
    }

    // 分割なし（または ffmpeg 未導入時）の一括処理
    const prompt = buildPrompt({
      glossary,
      partIndex: 1,
      totalParts: 1,
      chunkSeconds,
    });
    const transcript = await transcribeSingleFile(inputPath, originalMime, prompt);

    return res.json({
      transcript,
      mode: useChunking && !ffmpegAvailable ? '一括処理（ffmpeg未検出のためフォールバック）' : '一括処理',
    });
  } catch (error) {
    console.error('Transcription error:', error);
    const message = error instanceof Error ? error.message : '文字起こし中にエラーが発生しました。';
    return res.status(500).json({ error: message });
  } finally {
    safeUnlink(inputPath);
    safeUnlink(chunkWorkDir);
  }
});

app.listen(port, '127.0.0.1', () => {
  console.log(`文字起こしアプリ起動中: http://127.0.0.1:${port}`);
});
