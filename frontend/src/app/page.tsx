'use client';

import { useState } from 'react';
import ThemeList, { ThemeItem } from '@/components/ThemeList';
import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  const [selectedTheme, setSelectedTheme] = useState<ThemeItem | null>(null);

  const handleThemeSelect = (theme: ThemeItem) => {
    console.log('Selected theme:', theme);
    setSelectedTheme(theme);
  };

  const handleBackToThemes = () => {
    setSelectedTheme(null);
  };

  if (!selectedTheme) {
    return <ThemeList onThemeSelect={handleThemeSelect} />;
  }

  return (
    <div>
      <div className="chat-header-info">
        <button onClick={handleBackToThemes} className="back-button">
          ← テーマ選択に戻る
        </button>
        <div className="current-theme">
          <span className="theme-label">現在のテーマ:</span>
          <span className="theme-name">{selectedTheme.title}</span>
        </div>
      </div>
      <ChatInterface selectedTheme={selectedTheme} />
      
      <style jsx>{`
        .chat-header-info {
          background: #2563eb;
          color: white;
          padding: 0.75rem 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 1px solid #1d4ed8;
        }

        .back-button {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 0.25rem;
          cursor: pointer;
          font-size: 0.9rem;
          transition: background 0.2s;
        }

        .back-button:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .current-theme {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .theme-label {
          opacity: 0.8;
          font-size: 0.9rem;
        }

        .theme-name {
          font-weight: 600;
        }

        @media (max-width: 768px) {
          .chat-header-info {
            flex-direction: column;
            gap: 0.5rem;
            text-align: center;
          }
        }
      `}</style>
    </div>
  );
}