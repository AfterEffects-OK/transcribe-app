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
const envFilePath = path.join(process.cwd(), '.env');

// 実行中に更新可能なAPIキー保持変数
let currentApiKey = (process.env.GEMINI_API_KEY || '').trim();

function getAiClient() {
  if (!currentApiKey) {
    throw new Error('Gemini API キーが未設定です。画面右上の「API設定」から登録してください。');
  }
  return new GoogleGenAI({ apiKey: currentApiKey });
}

// .env ファイルの GEMINI_API_KEY を安全に更新・作成する
function saveApiKeyToEnv(newKey) {
  const sanitizedKey = newKey.replace(/[\r\n]/g, '').trim();
  let lines = [];

  if (fs.existsSync(envFilePath)) {
    const raw = fs.readFileSync(envFilePath, 'utf8');
    lines = raw.split(/\r?\n/);
  }

  let updated = false;
  lines = lines.map((line) => {
    if (/^\s*GEMINI_API_KEY\s*=/.test(line)) {
      updated = true;
      return `GEMINI_API_KEY=${sanitizedKey}`;
    }
    return line;
  });

  if (!updated) {
    lines.push(`GEMINI_API_KEY=${sanitizedKey}`);
  }

  if (!lines.some((line) => /^\s*PORT\s*=/.test(line))) {
    lines.push(`PORT=${port}`);
  }

  const content = lines.filter((l, idx, arr) => !(idx === arr.length - 1 && l === '')).join('\n') + '\n';
  fs.writeFileSync(envFilePath, content, 'utf8');
  currentApiKey = sanitizedKey;
  process.env.GEMINI_API_KEY = sanitizedKey;
}

const uploadDir = path.join(os.tmpdir(), 'gemini-transcribe-uploads');
fs.mkdirSync(uploadDir, { recursive: true });

const upload = multer({
  dest: uploadDir,
  limits: {
    fileSize: 4 * 1024 * 1024 * 1024,
  },
});

app.use(express.json());
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

// 設定状態の取得API
app.get('/api/status', async (_req, res) => {
  const ffmpegAvailable = await hasFfmpeg();
  const maskedKey = currentApiKey
    ? `${currentApiKey.slice(0, 6)}...${currentApiKey.slice(-4)}`
    : '';
  res.json({
    configured: Boolean(currentApiKey),
    maskedKey,
    ffmpegAvailable,
  });
});

// APIキーの接続テスト＆.env保存API
app.post('/api/settings', async (req, res) => {
  const apiKey = typeof req.body?.apiKey === 'string' ? req.body.apiKey.trim() : '';
  if (!apiKey) {
    return res.status(400).json({ error: 'APIキーを入力してください。' });
  }

  try {
    // 入力されたキーで接続テスト（軽量なモデル情報取得）
    const testClient = new GoogleGenAI({ apiKey });
    await testClient.models.get({ model: 'gemini-2.5-flash' });

    saveApiKeyToEnv(apiKey);
    const maskedKey = `${apiKey.slice(0, 6)}...${apiKey.slice(-4)}`;
    return res.json({
      ok: true,
      maskedKey,
      message: '接続テストに成功し、.env に保存しました。',
    });
  } catch (error) {
    console.error('API key verification failed:', error);
    return res.status(400).json({
      error: 'APIキーの検証に失敗しました。キーが正しいか、有効になっているか確認してください。',
    });
  }
});

// 音声/動画を30分（1800秒）単位のMP3チャンクに分割
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
async function uploadAndWaitActive(ai, filePath, mimeType) {
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

async function transcribeSingleFile(ai, filePath, mimeType, promptText) {
  const uploaded = await uploadAndWaitActive(ai, filePath, mimeType);
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
    const ai = getAiClient();
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
          const text = await transcribeSingleFile(ai, chunkPaths[i], 'audio/mp3', prompt);
          partsText.push(`=== パート ${i + 1} / ${chunkPaths.length} ===\n\n${text.trim()}`);
        }
        return res.json({
          transcript: partsText.join('\n\n---\n\n'),
          mode: `30分分割処理（全${chunkPaths.length}パート）`,
        });
      }
    }

    const prompt = buildPrompt({
      glossary,
      partIndex: 1,
      totalParts: 1,
      chunkSeconds,
    });
    const transcript = await transcribeSingleFile(ai, inputPath, originalMime, prompt);

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
  if (!currentApiKey) {
    console.log('※ Gemini API キーが未設定です。ブラウザ画面のセットアップから登録してください。');
  }
});
