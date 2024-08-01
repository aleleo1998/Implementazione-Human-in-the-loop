import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../css/Action.css';

const Actions = () => {
  const [actions, setActions] = useState([]);

  useEffect(() => {
    const fetchActions = async () => {
      try {
        const response = await axios.get('http://localhost:6040/get-actions');
        setActions(response.data);
      } catch (error) {
        console.error('Errore nel recupero delle azioni:', error);
      }
    };

    fetchActions();
  }, []);

  return (
    <div className="actions-container">
      <div className="actions-content">
        <h1>Actions</h1>
        <ul>
          {actions.map((action, index) => (
            <li key={index}>
              <strong>User:</strong> {action[0]}, <strong>Date:</strong> {action[1]}, <strong>Description:</strong> {action[2]}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Actions;
