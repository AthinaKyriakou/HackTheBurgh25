import React, { useState } from 'react';

const Conversation = ({ messages, onFeedbackSubmit }) => {
  const [feedbackText, setFeedbackText] = useState('');
  const [showFeedbackInput, setShowFeedbackInput] = useState(false);

  const handleThumbsUp = () => {
    alert('Thank you for your feedback!');
    setShowFeedbackInput(false); // Reset in case it was open
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

  return (
    <div className="conversation">
      {messages.map((msg, index) => (
        <div key={index} className={`message ${msg.role}`}>
          <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong> 
          <pre>{msg.content}</pre>
          {msg.role === 'assistant' && index === messages.length - 1 && (
            <div className="feedback">
              <button onClick={handleThumbsUp}>👍</button>
              <button onClick={handleThumbsDown}>👎</button>
              {showFeedbackInput && (
                <div>
                  <input
                    type="text"
                    value={feedbackText}
                    onChange={(e) => setFeedbackText(e.target.value)}
                    placeholder="Why didn't you like this?"
                  />
                  <button onClick={handleFeedbackSubmit}>Submit Feedback</button>
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default Conversation;