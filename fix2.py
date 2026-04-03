import pathlib

app_jsx = '''import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { sendChatMessage } from "./services/api.js";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (e) => {
    if (e) e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setMessages(prev => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendChatMessage(question);
      const answer = data.answer || data.response || "Không nhận được phản hồi từ hệ thống.";
      setMessages(prev => [...prev, { role: "assistant", content: answer, meta: data.citations?.length ? \Dựa trên \ tài liệu\ : "AI" }]);
    } catch (requestError) {
      console.error(requestError);
      setMessages(prev => [...prev, { role: "assistant", content: "Hiện tại tôi chưa lấy được kết quả từ máy chủ. Vui lòng thử lại sau." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* LEFT SIDEBAR */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">AI<span>|</span></div>
          <div>TRA CỨU LUẬT</div>
        </div>
        
        <button className="new-chat-btn" onClick={() => setMessages([])}>
          + Cuộc trò chuyện mới
        </button>

        <nav className="sidebar-nav">
          {["Văn bản Pháp Luật", "Tra cứu Văn bản", "Thủ tục hành chính", "Hỗ trợ"].map(item => (
            <div key={item} className="nav-item">
               <span style={{opacity: 0.5}}></span> {item}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="support-btn"> Liên hệ hỗ trợ</div>
          <div className="company-info">
            Một sản phẩm của
            <strong>CÔNG TY CỔ PHẦN DECOM STARS</strong>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">
        <header className="main-header">
          <div className="tabs">
            <div className="tab active">Chat với AI</div>
          </div>
          <div className="header-actions">
            <button className="pro-btn">Nâng cấp lên Pro </button>
            <div className="avatar">A</div>
          </div>
        </header>

        {/* RIGHT FLOATING CARD (HISTORY) */}
        <div className="history-card">
          <h3>Lịch sử trò chuyện</h3>
          <div className="history-item">Mẫu Đơn Ly Hôn</div>
          <div className="history-item">Hỗ Trợ Làm BGT</div>
        </div>

        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="hero-section">
              <div className="hero-logo">AI<span>|</span></div>
              <h1 className="hero-title">AI Tra cứu Luật có thể hỗ trợ gì cho bạn?</h1>
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((msg, idx) => (
                <div key={idx} className={\message \\}>
                  <div className="message-avatar">{msg.role === "user" ? "U" : "AI"}</div>
                  <div className="message-content">
                    {msg.role === "assistant" && <div className="message-meta">{msg.meta}</div>}
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                </div>
              ))}
              
              {loading && (
                <div className="message assistant">
                  <div className="message-avatar">AI</div>
                  <div className="message-content typing-dots">
                    <span/><span/><span/>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="input-wrapper">
          <form className="input-box" onSubmit={sendMessage}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Nhập câu hỏi của bạn tại đây..."
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
              }}
            />
            <div className="input-footer">
              <div className="input-badge">Standard</div>
              <div className="input-actions">
                <span className="char-count">{input.length}/2000</span>
                <button type="submit" className="send-btn" disabled={!input.trim() || loading}>
                  
                </button>
              </div>
            </div>
          </form>
          <div className="footer-note">
            Thông tin được tạo ra bằng AI. Hãy luôn cẩn trọng và sử dụng thông tin AI một cách có trách nhiệm.<br/>
            Xem thêm về <a href="#">Quyền riêng tư của bạn và ứng dụng AI Tra Cứu Luật.</a>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;'''

pathlib.Path(r'd:\AI_ENGINEERING\aichatbot-rag-law\frontend\src\App.jsx').write_text(app_jsx, encoding='utf-8')
