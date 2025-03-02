import React, { useState, useEffect, useRef } from 'react';

const Conversation = ({ messages, onFeedbackSubmit }) => {
  const [feedbackText, setFeedbackText] = useState('');
  const [showFeedbackInput, setShowFeedbackInput] = useState(false);
  const messagesEndRef = useRef(null);
  const conversationRef = useRef(null);

  const handleThumbsUp = () => {
    alert('Thank you for your feedback!');
    setShowFeedbackInput(false);
  };

  const handleThumbsDown = () => {
    setShowFeedbackInput(true);
  };

  const handleFeedbackSubmit = () => {
    if (feedbackText.trim()) {
      onFeedbackSubmit(feedbackText);
      setFeedbackText('');
      setShowFeedbackInput(false);
    }
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Format content with code blocks if needed
  const formatContent = (content) => {
    // Simple detection for JSON to add formatting
    try {
      JSON.parse(content);
      return (
        <pre className="code-block">
          {JSON.stringify(JSON.parse(content), null, 2)}
        </pre>
      );
    } catch (e) {
      // Regular content
      return <pre className="message-content">{content}</pre>;
    }
  };

  return (
    <div className="conversation-container" ref={conversationRef}>
      {messages.length === 0 ? (
        <div className="empty-state">
          <p>No messages yet. Submit a query to start the conversation.</p>
        </div>
      ) : (
        <div className="messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="message-header">
                <strong>{msg.role === 'user' ? 'You' : 'Assistant'}</strong>
                <span className="timestamp">{new Date().toLocaleTimeString()}</span>
              </div>
              <div className="message-body">
                {formatContent(msg.content)}
              </div>
              {msg.role === 'assistant' && index === messages.length - 1 && (
                <div className="feedback">
                  <button onClick={handleThumbsUp} className="feedback-btn" aria-label="Thumbs up">👍</button>
                  <button onClick={handleThumbsDown} className="feedback-btn" aria-label="Thumbs down">👎</button>
                  {showFeedbackInput && (
                    <div className="feedback-input">
                      <input
                        type="text"
                        value={feedbackText}
                        onChange={(e) => setFeedbackText(e.target.value)}
                        placeholder="Why didn't you like this?"
                      />
                      <button onClick={handleFeedbackSubmit} className="submit-feedback-btn">
                        Send
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}
    </div>
  );
};

export default Conversation;