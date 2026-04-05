import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

import { useChat } from './hooks/useChat.js'

function ProgressPanel({ ui }) {
  if (!ui.progressVisible) return null

  return (
    <div className="progress-panel" role="status" aria-live="polite">
      <div className="progress-top">
        <div className="progress-label">{ui.progressLabel}</div>
        <div className="progress-percent">{Math.round(ui.progressPercent)}%</div>
      </div>
      <div className="progress-rail" aria-hidden="true">
        <div className="progress-bar" style={{ width: `${ui.progressPercent}%` }} />
      </div>
      <div className="progress-steps" aria-hidden="true">
        {ui.steps.map((step, index) => (
          <div
            key={step.key}
            className={`progress-step ${index < ui.progressActiveIndex ? 'done' : index === ui.progressActiveIndex ? 'active' : ''}`}
          >
            <span className="progress-dot" />
            <span className="progress-text">{step.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Sources({ sources }) {
  if (!Array.isArray(sources) || sources.length === 0) return null
  return (
    <div className="citations">
      {sources.slice(0, 6).map((s) => (
        <span key={s.chunk_id || `${s.source_name}-${s.score}`} className="citation-tag" title={s.preview_text || ''}>
          {s.source_name || 'Nguồn'}
        </span>
      ))}
    </div>
  )
}

function App() {
  const { messages, loading, error, reset, cancel, send, ui } = useChat()
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const maxQueryChars = 2000

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading])

  const onSubmit = async (event) => {
    event.preventDefault()
    const text = input.trim()
    if (!text) return
    setInput('')
    await send(text)
  }

  return (
    <div className="app-layout">
      <aside className="sidebar" aria-label="Thanh điều hướng">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">
            AI<span>|</span>
          </div>
          <div>Tra cứu Luật (RAG)</div>
        </div>

        <button className="new-chat-btn" type="button" onClick={reset}>
          + Cuộc trò chuyện mới
        </button>

        <nav className="sidebar-nav" aria-label="Danh mục">
          {['Văn bản Pháp Luật', 'Tra cứu Văn bản', 'Thủ tục hành chính', 'Hỗ trợ'].map((item) => (
            <div key={item} className="nav-item">
              <span style={{ opacity: 0.45 }}>•</span> {item}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="support-btn">Liên hệ hỗ trợ</div>
          <div className="company-info">Thông tin hiển thị là mô phỏng UI (không gắn tên công ty bên thứ 3).</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="main-header">
          <div className="tabs">
            <div className="tab active">Chat với AI</div>
          </div>
          <div className="header-actions">
            <button className="pro-btn" type="button">
              Nâng cấp lên Pro
            </button>
            <div className="avatar" title="Tài khoản">
              U
            </div>
          </div>
        </header>

        <div className="chat-container">
          <ProgressPanel ui={ui} />

          {messages.length === 0 ? (
            <div className="hero-section">
              <div className="hero-logo">
                AI<span>|</span>
              </div>
              <h1 className="hero-title">Bạn cần tra cứu quy định pháp lý gì?</h1>
              <p className="hero-subtitle">Nhập câu hỏi tự nhiên. Hệ thống sẽ trả lời ngắn gọn kèm căn cứ liên quan.</p>
            </div>
          ) : (
            <div className="messages-list" aria-label="Luồng hội thoại">
              {messages.map((msg) => (
                <div key={msg.id} className={`message ${msg.role}`}>
                  <div className="message-avatar">{msg.role === 'user' ? 'U' : 'AI'}</div>
                  <div className="message-content">
                    {msg.role === 'assistant' && (
                      <div className="message-meta">
                        {msg.meta || 'AI pháp lý'}
                        {msg.status === 'streaming' ? <span className="stream-caret" aria-hidden="true" /> : null}
                      </div>
                    )}
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                    {msg.role === 'assistant' ? <Sources sources={msg.sources} /> : null}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message assistant" aria-label="Đang xử lý">
                  <div className="message-avatar">AI</div>
                  <div className="message-content typing-dots">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="input-wrapper">
          {error ? <div className="error-banner">{error}</div> : null}

          <form className="input-box" onSubmit={onSubmit}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Nhập câu hỏi của bạn tại đây… (tối đa 2000 ký tự)"
              maxLength={maxQueryChars}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  if (canSend) onSubmit(e)
                }
              }}
            />
            <div className="input-footer">
              <div className="input-badge">Standard</div>
              <div className="input-actions">
                <span className="char-count">{input.length}/{maxQueryChars}</span>
                {loading ? (
                  <button type="button" className="send-btn" onClick={cancel} title="Dừng">
                    ■
                  </button>
                ) : (
                  <button type="submit" className="send-btn" disabled={!canSend} title="Gửi">
                    ➔
                  </button>
                )}
              </div>
            </div>
          </form>
          <div className="footer-note">
            Thông tin được tạo ra bằng AI. Hãy luôn cẩn trọng và kiểm tra lại theo văn bản pháp luật gốc.
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
