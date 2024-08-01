import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import '../css/Login.css';

const Login = ({ setToken }) => {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleClick = async () => {
    try {
      const response = await axios.post('http://localhost:6040/ssi-login', {
        username,
        password
      });
      const token = response.data.Token;
      localStorage.setItem('token', token);
      setToken(token);
      alert(token);
      navigate('/training');  // Effettua il reindirizzamento dopo il login
    } catch (error) {
      console.error('Login error:', error);
      alert('Login con SSI fallito!');
    }
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <div className="login-text">
          Welcome, log in with SSI
        </div>
        <form>
          <div>
            <label htmlFor="username">Username:</label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div>
            <label htmlFor="password">Password:</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <div>
            <button type="button" onClick={handleClick}>
              Login in with SSI
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Login;
