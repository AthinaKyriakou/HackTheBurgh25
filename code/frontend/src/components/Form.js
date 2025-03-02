import React, { useState } from 'react';
import Select from 'react-select';

const degrees = [
  { value: 'Computer Science', label: 'Computer Science' },
  { value: 'Data Science', label: 'Data Science' },
  { value: 'Software Engineering', label: 'Software Engineering' },
];

const interestsOptions = [
  { value: 'AI', label: 'AI' },
  { value: 'Machine Learning', label: 'Machine Learning' },
  { value: 'Web Development', label: 'Web Development' },
  { value: 'Databases', label: 'Databases' },
  // Add more as needed
];

const Form = ({ onSubmit }) => {
  const [studentId, setStudentId] = useState('');
  const [degree, setDegree] = useState(null);
  const [semester, setSemester] = useState('1');
  const [credits, setCredits] = useState(0);
  const [interests, setInterests] = useState([]);
  const [freeText, setFreeText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = {
      studentId,
      degree: degree?.value || 'Not specified',
      semester,
      credits,
      interests: interests.map(i => i.value),
      additionalInfo: freeText || 'None'
    };
    onSubmit(JSON.stringify(formData));
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>Student ID:</label>
        <input
          type="text"
          value={studentId}
          onChange={(e) => setStudentId(e.target.value)}
          required
        />
      </div>
      <div>
        <label>Degree:</label>
        <Select
          options={degrees}
          value={degree}
          onChange={setDegree}
          placeholder="Select a degree"
        />
      </div>
      <div>
        <label>Semester:</label>
        <select value={semester} onChange={(e) => setSemester(e.target.value)}>
          <option value="1">1</option>
          <option value="2">2</option>
        </select>
      </div>
      <div>
        <label>Credits: {credits}</label>
        <input
          type="range"
          min="0"
          max="60"
          step="10"
          value={credits}
          onChange={(e) => setCredits(Number(e.target.value))}
        />
      </div>
      <div>
        <label>Interests:</label>
        <Select
          isMulti
          options={interestsOptions}
          value={interests}
          onChange={setInterests}
          placeholder="Select interests"
        />
      </div>
      <div>
        <label>Additional Info:</label>
        <textarea
          value={freeText}
          onChange={(e) => setFreeText(e.target.value)}
          rows="4"
          placeholder="Enter any additional information"
        />
      </div>
      <button type="submit">Submit</button>
    </form>
  );
};

export default Form;