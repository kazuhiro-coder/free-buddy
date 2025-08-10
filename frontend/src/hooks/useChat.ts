'use client';

import { useState, useCallback, useRef, useEffect } from 'react';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (message: string, onToken?: (token: string) => void, themeId?: number) => Promise<void>;
  clearMessages: () => void;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const useChat = (): UseChatReturn => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // アンマウント時に読み取りを中断
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        try { abortControllerRef.current.abort(); } catch {}
      }
    };
  }, []);

  const sendMessage = useCallback(async (
    message: string, 
    onToken?: (token: string) => void,
    themeId?: number
  ): Promise<void> => {
    if (!message.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);

    const now = Date.now();

    const userMessage: ChatMessage = {
      id: `${now}`,
      role: 'user',
      content: message,
      timestamp: new Date(),
    };

    const assistantMessageId = `${now + 1}`;
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);

    try {
      if (abortControllerRef.current) {
        try { abortControllerRef.current.abort(); } catch {}
      }
      abortControllerRef.current = new AbortController();

      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          user: 'user', 
          message,
          theme: themeId || 1
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Response body is not readable');

      const decoder = new TextDecoder();
      let assistantContent = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            let jsonData: any;
            try {
              jsonData = JSON.parse(line.slice(6));
            } catch {
              continue;
            }

            if (jsonData.error) {
              throw new Error(jsonData.error);
            }

            if (jsonData.content) {
              assistantContent += jsonData.content;
              setMessages(prev =>
                prev.map(msg =>
                  msg.id === assistantMessageId ? { ...msg, content: assistantContent } : msg
                )
              );
              if (onToken) onToken(jsonData.content);
            }

            if (jsonData.done) {
              break;
            }
          }
        }
      } finally {
        try { reader.releaseLock(); } catch {}
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        return;
      }
      const msg = err?.message || 'メッセージの送信中にエラーが発生しました';
      setError(msg);
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMessageId ? { ...m, content: `エラー: ${msg}` } : m
        )
      );
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [isLoading]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    if (abortControllerRef.current) {
      try { abortControllerRef.current.abort(); } catch {}
      abortControllerRef.current = null;
    }
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
  };
};
