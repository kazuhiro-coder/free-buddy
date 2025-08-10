'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useChat } from '@/hooks/useChat';
import { useSpeechRecognition } from '@/hooks/useSpeechRecognition';
import { useSpeechSynthesis } from '@/hooks/useSpeechSynthesis';
import { useTtsSettings } from '@/hooks/useTtsSettings';

interface ChatInterfaceProps {
  selectedTheme?: { id: number; title: string; description: string };
}

export default function ChatInterface({ selectedTheme }: ChatInterfaceProps) {
  const [textInput, setTextInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTtsActive, setIsTtsActive] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false); // ← 追加：設定UI開閉

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const speechQueueRef = useRef<string[]>([]);
  const isProcessingQueueRef = useRef<boolean>(false);
  const suppressTtsRef = useRef<boolean>(false);

  const { messages, isLoading, error, sendMessage, clearMessages } = useChat();
  const { 
    transcript, 
    isListening, 
    isSupported: isSpeechRecognitionSupported, 
    startListening, 
    stopListening,
    hasError: speechError,
    errorMessage: speechErrorMessage
  } = useSpeechRecognition();
  const { speak, isSpeaking, isSupported: isSpeechSynthesisSupported, stop: stopSpeaking } = useSpeechSynthesis();

  // 新規：TTS設定を利用
  const { settings: tts, setSettings: setTts, voices, refreshVoices, resolveVoice, testSpeak } = useTtsSettings();

  const SPEECH_TIMEOUT_MS = 30000;
  const SPEECH_GAP_MS = 200;
  const SENTENCE_END_RE = /[.!?]\s+|[.!?]$|\n/;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  useEffect(() => {
    if (transcript) setTextInput(transcript);
  }, [transcript]);

  useEffect(() => {
    return () => {
      try { window.speechSynthesis.cancel(); } catch {}
      speechQueueRef.current = [];
      isProcessingQueueRef.current = false;
      suppressTtsRef.current = true;
      setIsTtsActive(false);
    };
  }, []);

  const processSpeechQueue = useCallback(async () => {
    if (isProcessingQueueRef.current) return;

    isProcessingQueueRef.current = true;
    setIsTtsActive(true);

    try {
      while (!suppressTtsRef.current && speechQueueRef.current.length > 0) {
        const textToSpeak = speechQueueRef.current.shift();

        if (textToSpeak && isSpeechSynthesisSupported && textToSpeak.trim()) {
          await new Promise<void>((resolve) => {
            let resolved = false;
            const finish = () => { if (!resolved) { resolved = true; resolve(); } };

            const utterance = new window.SpeechSynthesisUtterance(textToSpeak.trim());
            const chosen = resolveVoice();
            if (chosen) utterance.voice = chosen;

            // ★ TTS設定を適用
            utterance.volume = tts.volume;
            utterance.rate   = tts.rate;
            utterance.pitch  = tts.pitch;
            utterance.lang   = chosen?.lang ?? 'en-US';

            const to = setTimeout(() => { try { window.speechSynthesis.cancel(); } catch {}; finish(); }, SPEECH_TIMEOUT_MS);
            utterance.onend = () => { clearTimeout(to); finish(); };
            utterance.onerror = () => { clearTimeout(to); finish(); };

            try { window.speechSynthesis.cancel(); } catch {}
            if (suppressTtsRef.current) { clearTimeout(to); finish(); return; }

            window.speechSynthesis.speak(utterance);
          });

          if (suppressTtsRef.current) break;
          await new Promise(r => setTimeout(r, SPEECH_GAP_MS));
        }
      }
    } finally {
      isProcessingQueueRef.current = false;
      setIsTtsActive(false);
    }
  }, [isSpeechSynthesisSupported, resolveVoice, tts.volume, tts.rate, tts.pitch]);

  const addToSpeechQueue = useCallback((text: string) => {
    if (suppressTtsRef.current) return;
    const t = text.trim();
    if (!t) return;
    speechQueueRef.current.push(t);
    if (!isProcessingQueueRef.current) processSpeechQueue();
  }, [processSpeechQueue]);

  const handleStopSpeaking = useCallback(() => {
    suppressTtsRef.current = true;
    stopSpeaking();
    try { window.speechSynthesis.cancel(); } catch {}
    speechQueueRef.current = [];
    isProcessingQueueRef.current = false;
    setIsTtsActive(false);
  }, [stopSpeaking]);

  const handleSendMessage = useCallback(async (message: string) => {
    if (!message.trim()) return;

    if (isListening) { stopListening(); setIsRecording(false); }
    setTextInput('');

    // Stop状態を解除して次の応答を読めるように
    suppressTtsRef.current = false;
    if (isSpeaking) stopSpeaking();
    try { window.speechSynthesis.cancel(); } catch {}
    speechQueueRef.current = [];
    isProcessingQueueRef.current = false;
    setIsTtsActive(false);

    let sentenceBuffer = '';

    await sendMessage(message, (token) => {
      sentenceBuffer += token;
      if (SENTENCE_END_RE.test(sentenceBuffer) || sentenceBuffer.length > 80) {
        const textToSpeak = sentenceBuffer.trim();
        if (textToSpeak) addToSpeechQueue(textToSpeak);
        sentenceBuffer = '';
      }
    }, selectedTheme?.id);

    if (sentenceBuffer.trim()) addToSpeechQueue(sentenceBuffer.trim());
  }, [addToSpeechQueue, isListening, isSpeaking, sendMessage, stopListening, stopSpeaking, selectedTheme]);

  const handleMicClick = useCallback(() => {
    if (isListening) {
      stopListening();
      setIsRecording(false);
    } else {
      setIsRecording(true);
      startListening((finalTranscript) => {
        if (finalTranscript.trim()) {
          stopListening();
          setIsRecording(false);
          handleSendMessage(finalTranscript);
        } else {
          setIsRecording(false);
        }
      });
    }
  }, [isListening, startListening, stopListening, handleSendMessage]);

  const handleTextSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (textInput.trim()) handleSendMessage(textInput);
  }, [textInput, handleSendMessage]);

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>🎤 VoiceChat with AI</h1>
        <div className="header-actions">
          <button onClick={() => setSettingsOpen(v => !v)} className="settings-button">
            {settingsOpen ? 'Close Settings' : 'Settings'}
          </button>
          <button onClick={clearMessages} className="clear-button">Clear Chat</button>
        </div>
      </div>

      {/* 簡易設定パネル（同ファイル内に配置） */}
      {settingsOpen && (
        <div className="settings-panel">
          <h2>🔧 TTS Settings</h2>

          <div className="row">
            <label>Voice</label>
            <div className="voice-row">
              <select
                value={tts.voiceName ?? ''}
                onChange={(e) => setTts({ voiceName: e.target.value || null })}
              >
                <option value="">(Auto)</option>
                {voices.map(v => (
                  <option key={`${v.name}_${v.lang}`} value={v.name}>
                    {v.name} {v.lang ? `(${v.lang})` : ''}
                  </option>
                ))}
              </select>
              <button type="button" onClick={refreshVoices}>↻ Refresh</button>
            </div>
          </div>

          <div className="row">
            <label>Volume: {tts.volume.toFixed(2)}</label>
            <input type="range" min={0} max={1} step={0.01}
              value={tts.volume}
              onChange={(e) => setTts({ volume: Number(e.target.value) })}
            />
          </div>

          <div className="row">
            <label>Rate: {tts.rate.toFixed(2)}</label>
            <input type="range" min={0.5} max={2} step={0.05}
              value={tts.rate}
              onChange={(e) => setTts({ rate: Number(e.target.value) })}
            />
          </div>

          <div className="row">
            <label>Pitch: {tts.pitch.toFixed(2)}</label>
            <input type="range" min={0} max={2} step={0.05}
              value={tts.pitch}
              onChange={(e) => setTts({ pitch: Number(e.target.value) })}
            />
          </div>

          <div className="row">
            <label>Sample</label>
            <div className="sample-row">
              <input
                defaultValue="This is a sample sentence."
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    const v = (e.target as HTMLInputElement).value;
                    testSpeak(v);
                  }
                }}
              />
              <button type="button" onClick={() => {
                const el = document.querySelector('.sample-row input') as HTMLInputElement | null;
                testSpeak(el?.value || 'This is a sample sentence.');
              }}>▶︎ Test</button>
            </div>
          </div>
        </div>
      )}

      <div className="messages-container">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
          >
            <div className="message-role">
              {message.role === 'user' ? '👤 You' : '🤖 AI'}
            </div>
            <div className="message-content">{message.content}</div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message assistant-message">
            <div className="message-role">🤖 AI</div>
            <div className="message-content typing">Thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-section">
        {error && <div className="error-message">❌ {error}</div>}
        {speechError && <div className="error-message">❌ {speechErrorMessage}</div>}

        <div className="voice-controls">
          {isSpeechRecognitionSupported ? (
            <button
              onClick={handleMicClick}
              className={`mic-button ${isListening ? 'recording' : ''}`}
              disabled={isLoading}
            >
              {isListening ? '🎙️ Stop Recording' : '🎤 Start Recording'}
            </button>
          ) : (
            <div className="no-support-message">⚠️ Speech recognition is not supported in this browser</div>
          )}

          {(isSpeaking || isTtsActive) && (
            <button onClick={handleStopSpeaking} className="stop-speaking-button">
              🔇 Stop Speaking
            </button>
          )}
        </div>

        <div className="text-input-section">
          <form onSubmit={handleTextSubmit} className="text-form">
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder={isSpeechRecognitionSupported ? "Speak or type your message..." : "Type your message..."}
              className="text-input"
              disabled={isLoading}
            />
            <button type="submit" className="send-button" disabled={isLoading || !textInput.trim()}>
              Send
            </button>
          </form>
        </div>

        <div className="status-indicators">
          <div className={`status-indicator ${isSpeechRecognitionSupported ? 'supported' : 'not-supported'}`}>
            STT: {isSpeechRecognitionSupported ? '✅' : '❌'}
          </div>
          <div className={`status-indicator ${isSpeechSynthesisSupported ? 'supported' : 'not-supported'}`}>
            TTS: {isSpeechSynthesisSupported ? '✅' : '❌'}
          </div>
          <div className={`status-indicator ${isListening ? 'active' : ''}`}>
            Listening: {isListening ? '🟢' : '⚫'}
          </div>
          <div className={`status-indicator ${(isSpeaking || isTtsActive) ? 'active' : ''}`}>
            Speaking: {(isSpeaking || isTtsActive) ? '🟢' : '⚫'}
          </div>
        </div>
      </div>

      <style jsx>{`
        .chat-container { display: flex; flex-direction: column; height: 100vh; max-width: 800px; margin: 0 auto; background: #f5f5f5; }
        .chat-header { background: #2563eb; color: white; padding: 1rem; display: flex; justify-content: space-between; align-items: center; }
        .header-actions { display: flex; gap: .5rem; }
        .clear-button { background: #ef4444; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; }
        .settings-button { background: #0ea5e9; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; }

        .settings-panel { background: #fff; border-top: 1px solid #e5e5e5; padding: 1rem; }
        .settings-panel h2 { margin: 0 0 .75rem 0; }
        .row { display: grid; gap: .25rem; margin-bottom: .75rem; }
        .voice-row { display: flex; gap: .5rem; align-items: center; }
        .sample-row { display: flex; gap: .5rem; align-items: center; }
        select, input[type="text"] { padding: .5rem; border: 1px solid #d1d5db; border-radius: 6px; }
        input[type="range"] { width: 100%; }

        .messages-container { flex: 1; overflow-y: auto; padding: 1rem; background: white; }
        .message { margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; max-width: 80%; }
        .user-message { background: #dbeafe; margin-left: auto; }
        .assistant-message { background: #f3f4f6; margin-right: auto; }
        .message-role { font-weight: bold; margin-bottom: 0.5rem; font-size: 0.875rem; }
        .message-content { margin-bottom: 0.5rem; white-space: pre-wrap; }
        .message-timestamp { font-size: 0.75rem; opacity: 0.6; }
        .typing { font-style: italic; opacity: 0.7; }

        .input-section { background: white; border-top: 1px solid #e5e5e5; padding: 1rem; }
        .error-message { background: #fef2f2; color: #dc2626; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 1rem; }

        .voice-controls { display: flex; gap: 1rem; margin-bottom: 1rem; justify-content: center; }
        .mic-button { background: #10b981; color: white; border: none; padding: 1rem 2rem; border-radius: 2rem; cursor: pointer; font-size: 1rem; transition: all 0.2s; }
        .mic-button.recording { background: #ef4444; animation: pulse 2s infinite; }
        .stop-speaking-button { background: #f59e0b; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; }

        .no-support-message { color: #dc2626; text-align: center; font-size: 0.875rem; }
        .text-input-section { margin-bottom: 1rem; }
        .text-form { display: flex; gap: 0.5rem; }
        .text-input { flex: 1; padding: 0.75rem; border: 1px solid #d1d5db; border-radius: 0.375rem; font-size: 1rem; }
        .send-button { background: #2563eb; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.375rem; cursor: pointer; }
        .send-button:disabled { background: #9ca3af; cursor: not-allowed; }

        .status-indicators { display: flex; gap: 1rem; justify-content: center; font-size: 0.875rem; }
        .status-indicator { padding: 0.25rem 0.5rem; border-radius: 0.25rem; background: #f3f4f6; }
        .status-indicator.supported { background: #d1fae5; }
        .status-indicator.not-supported { background: #fef2f2; }
        .status-indicator.active { background: #dcfce7; }

        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.05); } }
        @media (max-width: 768px) {
          .voice-controls { flex-direction: column; }
          .status-indicators { flex-wrap: wrap; }
          .message { max-width: 95%; }
        }
      `}</style>
    </div>
  );
}
