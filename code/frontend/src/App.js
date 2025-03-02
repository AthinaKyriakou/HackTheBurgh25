import React, { useState } from 'react';
import Form from './components/Form';
import Conversation from './components/Conversation';
import './App.css'; // Optional: for styling

const App = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleFormSubmit = async (formattedMessage) => {
    const userMessage = { role: 'user', content: formattedMessage };
    setMessages([userMessage]); // Reset conversation for new form submission
    setIsLoading(true);
    await sendMessage([userMessage]);
    setIsLoading(false);
  };

  const handleFeedbackSubmit = async (feedbackText) => {
    const feedbackMessage = { role: 'user', content: feedbackText };
    const updatedMessages = [...messages, feedbackMessage];
    setMessages(updatedMessages);
    setIsLoading(true);
    await sendMessage(updatedMessages);
    setIsLoading(false);
  };

  const sendMessage = async (messagesToSend) => {
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: messagesToSend }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = { role: 'assistant', content: '' };
      setMessages([...messagesToSend, assistantMessage]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        assistantMessage.content += chunk;
        setMessages([...messagesToSend, { ...assistantMessage }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages([...messagesToSend, { role: 'assistant', content: 'Error occurred' }]);
    }
  };

  return (
    <div className="App">
      <h1>Student Query System</h1>
      <Form onSubmit={handleFormSubmit} />
      <Conversation messages={messages} onFeedbackSubmit={handleFeedbackSubmit} />
      {isLoading && <div className="spinner">Loading...</div>}
    </div>
  );
};

export default App;