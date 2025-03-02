import React, { useState } from 'react';
import Select from 'react-select';

const degrees = [
  { value: 'Computer Science', label: 'Computer Science' },
  { value: 'Data Science', label: 'Data Science' },
  { value: 'Software Engineering', label: 'Software Engineering' },
  { value: 'Artificial Intelligence', label: 'Artificial Intelligence' },
  { value: 'Cybersecurity', label: 'Cybersecurity' },
];

const interestsOptions = [
  { value: 'AI', label: 'AI' },
  { value: 'Machine Learning', label: 'Machine Learning' },
  { value: 'Web Development', label: 'Web Development' },
  { value: 'Databases', label: 'Databases' },
  { value: 'Mobile Development', label: 'Mobile Development' },
  { value: 'Cloud Computing', label: 'Cloud Computing' },
  { value: 'Cybersecurity', label: 'Cybersecurity' },
  { value: 'Data Science', label: 'Data Science' },
  { value: 'UI/UX Design', label: 'UI/UX Design' },
];

const Form = ({ onSubmit }) => {
  const [degree, setDegree] = useState(null);
  const [semester, setSemester] = useState('1');
  const [year, setYear] = useState('1');
  const [credits, setCredits] = useState(0);
  const [interests, setInterests] = useState([]);
  const [freeText, setFreeText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = {
      degree: degree?.value || 'Not specified',
      year: year,
      semester,
      credits,
      interests: interests.map(i => i.value),
      additionalInfo: freeText || 'None'
    };
    onSubmit(formData);
  };

  return (
    <form className="modern-form" onSubmit={handleSubmit}>
      <h2>Student Information</h2>
      
      <div className="form-group">
        <label htmlFor="year">Year of Study:</label>
        <select 
          id="year"
          value={year} 
          onChange={(e) => setYear(e.target.value)}
        >
          {[1, 2, 3, 4, 5, 6, 7].map(num => (
            <option key={num} value={num}>{num}</option>
          ))}
        </select>
      </div>
      
      <div className="form-group">
        <label htmlFor="degree">Degree Program:</label>
        <Select
          inputId="degree"
          className="react-select-container"
          classNamePrefix="react-select"
          options={degrees}
          value={degree}
          onChange={setDegree}
          placeholder="Select a degree"
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="semester">Semester:</label>
        <select 
          id="semester"
          value={semester} 
          onChange={(e) => setSemester(e.target.value)}
        >
          {[1, 2].map(num => (
            <option key={num} value={num}>{num}</option>
          ))}
        </select>
      </div>
      
      <div className="form-group">
        <label htmlFor="credits">
          Credits: <span className="value-display">{credits}</span>
        </label>
        <input
          id="credits"
          type="range"
          min="0"
          max="60"
          step="10"
          value={credits}
          onChange={(e) => setCredits(Number(e.target.value))}
        />
        <div className="range-labels">
          <span>0</span>
          <span>60</span>
        </div>
      </div>
      
      <div className="form-group">
        <label htmlFor="interests">Areas of Interest:</label>
        <Select
          inputId="interests"
          isMulti
          className="react-select-container"
          classNamePrefix="react-select"
          options={interestsOptions}
          value={interests}
          onChange={setInterests}
          placeholder="Select your interests"
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="freeText">Additional Information:</label>
        <textarea
          id="freeText"
          value={freeText}
          onChange={(e) => setFreeText(e.target.value)}
          rows="4"
          placeholder="Enter any questions or additional information"
        />
      </div>
      
      <button type="submit" className="submit-btn">Submit Query</button>
    </form>
  );
};

export default Form;