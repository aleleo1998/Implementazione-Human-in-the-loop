import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';
import Login from './pages/Login';
import Training from './pages/Training';
import Actions from './pages/Actions'

const App = () => {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [isValid, setIsValid] = useState(false);
  const [loading, setLoading] = useState(true);


  useEffect(() => {
    const checkTokenValidity = async () => {
      if (token) {
        try {
          const response = await axios.post('http://localhost:6040/verify-token', { token });
          if (response.data.valid) {
            setIsValid(true);
          } else {
            localStorage.removeItem('token');
            setToken(null);
          }
        } catch (error) {
          console.error('Token verification error:', error);
          localStorage.removeItem('token');
          setToken(null);
        }
      }
      setLoading(false);
    };

    checkTokenValidity();
  }, [token]);

  useEffect(() => {
    if (token) {
      localStorage.setItem('token', token);
    }
  }, [token]);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <Router>
      <Routes>
        <Route path="/" element={isValid ? <Navigate to="/training" /> : <Login setToken={setToken} />} />
        <Route path="/training" element={isValid ? <Training /> : <Navigate to="/" />} />
        <Route path="/actions" element={isValid? <Actions /> : <Navigate to="/" />}/>
      </Routes>
    </Router>
  );
};

export default App;
