import React, { useState } from 'react';
import Form from './components/Form';
import Conversation from './components/Conversation';
import './App.css';
import logo from './logo.png';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleFormSubmit = async (formattedMessage) => {
    const userMessage = { role: 'user', content: formattedMessage };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    await sendMessage([...messages, userMessage]);
    setIsLoading(false);
  };

  const handleFeedbackSubmit = async (feedbackText) => {
    const feedbackMessage = { role: 'user', content: feedbackText };
    setMessages((prev) => [...prev, feedbackMessage]);
    setIsLoading(true);
    await sendMessage([...messages, feedbackMessage]);
    setIsLoading(false);
  };

  const sendMessage = async (messagesToSend) => {
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: messagesToSend }),
      });
      
      if (!response.ok) throw new Error('Network response was not ok');
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = { role: 'assistant', content: '' };
      
      // Add the initial empty assistant message to the state
      setMessages((prevMessages) => [...messagesToSend, assistantMessage]);
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        assistantMessage.content += chunk;
        
        // Update the message state with the new content
        setMessages((prevMessages) => {
          const updatedMessages = [...prevMessages];
          updatedMessages[updatedMessages.length - 1] = { ...assistantMessage };
          return updatedMessages;
        });
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages((prevMessages) => [
        ...messagesToSend, 
        { role: 'assistant', content: 'Error occurred while processing your request. Please try again.' }
      ]);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <div className="banner-container">
            <img 
              src={logo} 
              alt="Student Query System Logo" 
              className="header-banner" 
            />
          </div>
          <h1>Student Study Planner</h1>
        </div>
      </header>
      <main className="app-main">
        <div className="form-section">
          <Form onSubmit={handleFormSubmit} />
        </div>
        <div className="chat-section">
          <Conversation messages={messages} onFeedbackSubmit={handleFeedbackSubmit} />
          {isLoading && <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Processing your request...</span>
          </div>}
        </div>
      </main>
    </div>
  );
};

export default App;