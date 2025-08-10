'use client';

import React, { useState } from 'react';

interface CreateThemeFormProps {
  onSubmit: (theme: { title: string; description: string; system_prompt: string }) => Promise<void>;
  onCancel: () => void;
  isLoading?: boolean;
  error?: string | null;
}

export default function CreateThemeForm({ onSubmit, onCancel, isLoading = false, error }: CreateThemeFormProps) {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    system_prompt: ''
  });
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.title.trim() || !formData.description.trim()) {
      return;
    }

    // system_promptが空の場合はデフォルトを設定
    const defaultSystemPrompt = `You are a friendly conversation partner discussing ${formData.title.toLowerCase()}. Keep responses conversational, engaging, and within 1-2 sentences. Ask follow-up questions to maintain the conversation flow. Respond in English only.`;
    
    await onSubmit({
      ...formData,
      system_prompt: formData.system_prompt.trim() || defaultSystemPrompt
    });
  };

  const handleChange = (field: keyof typeof formData) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormData(prev => ({
      ...prev,
      [field]: e.target.value
    }));
  };

  const isFormValid = formData.title.trim() && formData.description.trim();

  return (
    <div className="form-overlay">
      <div className="form-container">
        <div className="form-header">
          <h2>🎨 新しい会話テーマを作成</h2>
          <button onClick={onCancel} className="close-button" type="button">
            ×
          </button>
        </div>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="theme-form">
          <div className="form-group">
            <label htmlFor="title" className="form-label">
              テーマタイトル *
            </label>
            <input
              id="title"
              type="text"
              value={formData.title}
              onChange={handleChange('title')}
              placeholder="例: カフェでの注文、空港でのチェックイン"
              className="form-input"
              maxLength={100}
              disabled={isLoading}
              required
            />
            <div className="char-count">{formData.title.length}/100</div>
          </div>

          <div className="form-group">
            <label htmlFor="description" className="form-label">
              説明 *
            </label>
            <textarea
              id="description"
              value={formData.description}
              onChange={handleChange('description')}
              placeholder="このテーマでどのような会話をするか説明してください"
              className="form-textarea"
              maxLength={300}
              rows={3}
              disabled={isLoading}
              required
            />
            <div className="char-count">{formData.description.length}/300</div>
          </div>

          <div className="advanced-section">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="advanced-toggle"
              disabled={isLoading}
            >
              ⚙️ 高度な設定 {showAdvanced ? '▼' : '▶'}
            </button>
            
            {showAdvanced && (
              <div className="advanced-content">
                <div className="form-group">
                  <label htmlFor="system_prompt" className="form-label">
                    システムプロンプト（任意）
                  </label>
                  <textarea
                    id="system_prompt"
                    value={formData.system_prompt}
                    onChange={handleChange('system_prompt')}
                    placeholder="空欄の場合は自動生成されます"
                    className="form-textarea"
                    maxLength={500}
                    rows={4}
                    disabled={isLoading}
                  />
                  <div className="char-count">{formData.system_prompt.length}/500</div>
                  <div className="form-hint">
                    AIの話し方や役割を具体的に指定できます。空欄の場合は自動的に適切なプロンプトが生成されます。
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="form-actions">
            <button
              type="button"
              onClick={onCancel}
              className="cancel-button"
              disabled={isLoading}
            >
              キャンセル
            </button>
            <button
              type="submit"
              className="submit-button"
              disabled={isLoading || !isFormValid}
            >
              {isLoading ? '作成中...' : 'テーマを作成'}
            </button>
          </div>
        </form>
      </div>

      <style jsx>{`
        .form-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 1rem;
        }

        .form-container {
          background: white;
          border-radius: 1rem;
          width: 100%;
          max-width: 600px;
          max-height: 90vh;
          overflow-y: auto;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }

        .form-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1.5rem;
          border-bottom: 1px solid #e5e5e5;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-radius: 1rem 1rem 0 0;
        }

        .form-header h2 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 600;
        }

        .close-button {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: none;
          width: 2rem;
          height: 2rem;
          border-radius: 50%;
          cursor: pointer;
          font-size: 1.2rem;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.2s;
        }

        .close-button:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .theme-form {
          padding: 1.5rem;
        }

        .form-group {
          margin-bottom: 1.5rem;
        }

        .form-label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 500;
          color: #374151;
        }

        .form-input, .form-textarea {
          width: 100%;
          padding: 0.75rem;
          border: 2px solid #e5e7eb;
          border-radius: 0.5rem;
          font-size: 1rem;
          transition: border-color 0.2s;
          font-family: inherit;
          box-sizing: border-box;
        }

        .form-input:focus, .form-textarea:focus {
          outline: none;
          border-color: #667eea;
          box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .form-textarea {
          resize: vertical;
          min-height: 80px;
        }

        .char-count {
          text-align: right;
          font-size: 0.75rem;
          color: #6b7280;
          margin-top: 0.25rem;
        }

        .advanced-section {
          margin-bottom: 1.5rem;
        }

        .advanced-toggle {
          background: none;
          border: none;
          color: #667eea;
          cursor: pointer;
          font-size: 0.9rem;
          padding: 0.5rem 0;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .advanced-content {
          margin-top: 1rem;
          padding: 1rem;
          background: #f9fafb;
          border-radius: 0.5rem;
          border: 1px solid #e5e7eb;
        }

        .form-hint {
          font-size: 0.8rem;
          color: #6b7280;
          margin-top: 0.5rem;
          line-height: 1.4;
        }

        .error-message {
          background: #fef2f2;
          color: #dc2626;
          padding: 0.75rem;
          border-radius: 0.5rem;
          margin-bottom: 1rem;
          border: 1px solid #fecaca;
        }

        .form-actions {
          display: flex;
          gap: 1rem;
          justify-content: flex-end;
          padding-top: 1rem;
          border-top: 1px solid #e5e5e5;
        }

        .cancel-button, .submit-button {
          padding: 0.75rem 1.5rem;
          border-radius: 0.5rem;
          font-size: 1rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          border: none;
        }

        .cancel-button {
          background: #f3f4f6;
          color: #374151;
        }

        .cancel-button:hover:not(:disabled) {
          background: #e5e7eb;
        }

        .submit-button {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
        }

        .submit-button:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 10px 25px -5px rgba(102, 126, 234, 0.3);
        }

        .submit-button:disabled, .cancel-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }

        @media (max-width: 768px) {
          .form-overlay {
            padding: 0.5rem;
          }

          .form-container {
            max-height: 95vh;
          }

          .form-header {
            padding: 1rem;
          }

          .form-header h2 {
            font-size: 1.25rem;
          }

          .theme-form {
            padding: 1rem;
          }

          .form-actions {
            flex-direction: column-reverse;
          }

          .cancel-button, .submit-button {
            width: 100%;
          }
        }
      `}</style>
    </div>
  );
}