import React, { useState, useEffect, useRef } from 'react';
import { Send, AlertCircle, RefreshCcw, CheckCircle2, ChevronRight, Bot, HelpCircle, ArrowRight } from 'lucide-react';
import Markdown from 'markdown-to-jsx';

const API_BASE = '/api';

export default function App() {
  const [step, setStep] = useState('welcome'); // welcome, wizard, chat
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [currentQIdx, setCurrentQIdx] = useState(0);
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [checklist, setChecklist] = useState([]);
  const [progress, setProgress] = useState(0);

  const chatEndRef = useRef(null);

  useEffect(() => {
    if (step === 'chat') {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, step]);

  const startOnboarding = async () => {
    try {
      const res = await fetch(`${API_BASE}/onboarding/setup-questions`);
      const data = await res.json();
      setQuestions(data.questions);
      setStep('wizard');
    } catch (err) {
      console.error("Failed to load questions", err);
    }
  };

  const handleAnswer = (qid, val) => {
    setAnswers(prev => ({ ...prev, [qid]: val }));
    if (currentQIdx < questions.length - 1) {
      setCurrentQIdx(currentQIdx + 1);
    } else {
      finalizeSetup();
    }
  };

  const finalizeSetup = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/onboarding/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers })
      });
      const data = await res.json();
      setSession(data.session_id);
      setChecklist(data.checklist);
      setStep('chat');
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        text: `Chào bạn! Mình là trợ lý Onboarding. Chúc mừng bạn đã gia nhập VinUni với vai trò **${answers.role}**. Mình đã chuẩn bị một checklist cho bạn bên trái. Bạn cần hỗ trợ gì không?`
      }]);
    } catch (err) {
      console.error("Failed to init", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = { id: Date.now(), role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: input,
          session_id: session,
          role: answers.role === 'Tân sinh viên' ? 'freshman_student' : 'staff'
        })
      });
      const data = await res.json();
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        text: data.answer,
        citations: data.citations
      }]);

      // Update checklist progress
      const checkRes = await fetch(`${API_BASE}/onboarding/checklist/${session}`);
      const checkData = await checkRes.json();
      setChecklist(checkData.checklist);
      setProgress(checkData.progress);
    } catch (err) {
      console.error("Chat failed", err);
    } finally {
      setLoading(false);
    }
  };

  const handleAction = async (type) => {
    const feedback = prompt(`Bạn muốn ${type === 'report' ? 'báo sai' : 'chuyển bộ phận'} về vấn đề gì?`);
    if (!feedback) return;

    const endpoint = type === 'report' ? 'report-error' : 'transfer';
    await fetch(`${API_BASE}/actions/${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: session, action: type, feedback })
    });
    alert("Yêu cầu của bạn đã được gửi!");
  };

  if (step === 'welcome') {
    return (
      <div className="glass-card welcome-card animate-fade-in">
        <div className="welcome-icon-wrap">
          <Bot size={48} className="icon-primary" />
        </div>
        <h1 className="welcome-title">VinUni Onboarding</h1>
        <p className="welcome-subtitle">
          Chào mừng bạn đến với VinUniversity! Hãy bắt đầu hành trình của bạn với sự hỗ trợ của trợ lý AI thông minh.
        </p>
        <button className="btn-primary btn-lg" onClick={startOnboarding}>
          Bắt đầu ngay <ArrowRight size={22} />
        </button>
      </div>
    );
  }

  if (step === 'wizard') {
    const q = questions[currentQIdx];
    return (
      <div className="glass-card wizard-card animate-fade-in">
        <div className="wizard-step">Bước {currentQIdx + 1} / {questions.length}</div>
        <h2 className="wizard-question">{q?.text}</h2>
        <div className="wizard-options">
          {q?.options.map(opt => (
            <button
              key={opt}
              className="btn-secondary wizard-option-btn"
              onClick={() => handleAnswer(q.id, opt)}
            >
              {opt}
              <ChevronRight size={18} className="wizard-option-icon" />
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell animate-fade-in">
      {/* Sidebar Checklist */}
      <div className="glass-card sidebar-panel">
        <div className="sidebar-header">
          <h3 className="sidebar-title">Checklist của bạn</h3>
          <div className="sidebar-subtitle">Thực hiện {progress.toFixed(0)}% các bước</div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
        <div className="sidebar-tasks">
          {checklist.map(task => (
            <div key={task.id} className="task-item">
              {task.status === 'completed' ? <CheckCircle2 size={18} className="task-icon-done" /> : <div className="task-icon-pending" />}
              <div className="task-content">
                <div className="task-title">{task.title}</div>
                {task.link && <a href={task.link} target="_blank" className="task-link" rel="noreferrer">Truy cập ngay</a>}
              </div>
            </div>
          ))}
        </div>
        <div className="sidebar-actions">
          <button className="action-btn action-btn-report" onClick={() => handleAction('report')}>
            <AlertCircle size={14} /> Báo sai
          </button>
          <button className="action-btn action-btn-transfer" onClick={() => handleAction('transfer')}>
            <RefreshCcw size={14} /> Chuyển bộ phận
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="glass-card chat-panel">
        <div className="chat-header">
          <div className="chat-header-main">
            <div className="chat-avatar">
              <Bot size={20} />
            </div>
            <div>
              <div className="chat-title">Onboarding Assistant</div>
              <div className="chat-status">
                <span className="chat-status-dot"></span> Trực tuyến
              </div>
            </div>
          </div>
        </div>

        <div className="chat-messages">
          {messages.map(m => (
            <div key={m.id} className={`message-row ${m.role === 'user' ? 'is-user' : 'is-assistant'}`}>
              <div className={`message-bubble ${m.role === 'user' ? 'message-user' : 'message-assistant'}`}>
                <div className="message-markdown">
                  <Markdown>{m.text}</Markdown>
                </div>
                {m.citations && m.citations.length > 0 && (
                  <div className="citation-list">
                    {m.citations.map(c => (
                      <span key={c.id} className="citation-chip">
                        <HelpCircle size={10} /> {c.id}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message-row is-assistant">
              <div className="typing-bubble">
                <span className="typing-dot"></span>
                <span className="typing-dot"></span>
                <span className="typing-dot"></span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="chat-input-form">
          <input
            type="text"
            className="chat-input"
            placeholder="Hỏi mình bất cứ điều gì..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button type="submit" className="chat-send-btn">
            <Send size={24} />
          </button>
        </form>
      </div>
    </div>
  );
}
