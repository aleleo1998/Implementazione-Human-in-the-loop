
import React, { useEffect, useState } from 'react';
import '../css/Training.css'
import io from 'socket.io-client';

function Dot({ id, color, info }) {
  const [maxHeight, setMaxHeight] = useState(0);
  const infoLines = info.split('\n');

  useEffect(() => {
    const elements = document.querySelectorAll('.train-info');
    let max = 0;
    elements.forEach(element => {
      max = Math.max(max, element.scrollHeight);
    });
    setMaxHeight(max);
  }, [info]);

  return (
    <div className="dotContainer">
      <div className="dot" style={{ backgroundColor: color }} />
      <div className="dot-info">Client: {id + 1}</div>
      <div className="train-info" style={{ height: maxHeight }}>
        
      <div className='round-train-info'>
        {infoLines.map((line, index) => (
           line.startsWith("Round") ? (
            <div key={index}>
                <h3>{line}</h3>
            </div>
        ) : (
            <div key={index}>{line}</div>
        )
        ))}
      
      </div>

      </div>
    </div>
  );
}
function App() {
  const [dots, setDots] = useState(Array(4).fill({ color: 'red', info: '' }));
  const [buttonClicked, setButtonClicked] = useState(false);
  const [buttonServer, setButtonServer] = useState(false);
  const [sockets, setSockets] = useState([]);
  const [serverResults, setServerResults] = useState('');
  const [serverAggregate, setAggregate] = useState(false)
  const [serverDownlaod, setDownload] = useState(false)
  const [clientTraining, setClientTraining]=useState([])
  const [listModels, setListModels]=useState([])
  const [selectedId, setSelectedId] = useState(null);
  const [startRound]=useState(Array(4).fill(1))
  const [round,setRound]=useState(1)
  
  
 
  useEffect(() => {
    const newSockets = [];
  
   
    for (let i = 0; i <= 3; i++) {
      const socket = io(`http://127.0.0.1:${6001 + i}`);
      socket.on('data', data => {
        if (startRound[i]===1){
          data="Round "+round+"\n"+data
          startRound[i]=0
        }
        setDots(prevDots => {

         
          const newDots = [...prevDots];
          newDots[i] = {
            ...newDots[i],
            info: prevDots[i].info ? prevDots[i].info + '\n' + data : data // Aggiungi i nuovi dati su una nuova riga utilizzando <br>
          };
          return newDots;
        });
      });

     


      socket.on('startTraining', () => {
       
        setDots(prevDots => {
          
          const newDots = [...prevDots];
          newDots[i] = {
            ...newDots[i],
            color: 'green' // Cambia il colore del dot quando riceve l'evento 'starttraining'
          };
          return newDots;
        });
      });


      socket.on('stopTraining', () => {
        setDots(prevDots => {
          const newDots = [...prevDots];
          newDots[i] = {
            ...newDots[i],
            color: 'red' // Cambia il colore del dot quando riceve l'evento 'starttraining'
          };
          return newDots;
        });
      });

      newSockets.push(socket);
    }

    setSockets(newSockets);

    return () => {
      // Chiudi le connessioni quando il componente viene smontato
      newSockets.forEach(socket => socket.disconnect());
    };
  }, [round, startRound]);

  
      
  


  
  const startSimulation = () => {
      // Connessione ai server utilizzando Socket.IO
      setButtonClicked(true)
      sockets.forEach(socket => {    
        // Invia un messaggio a ciascun server
        socket.emit('startsimulation');
      });
    }
    
    
    const startserver = () => {
      const socketServer = io(`http://127.0.0.1:6020`);
      socketServer.emit('startserver');
      socketServer.emit('setUsername',localStorage.getItem('token'))
      socketServer.on('serverResults', newData => {
        setServerResults(prevData => prevData + '\n' + newData); // Aggiungi i nuovi dati ai risultati esistenti
      });
     

      socketServer.on('startAggragate' ,() => {
        for (let i = 0; i <= 3; i++) {
          startRound[i]=1
        }
        setRound(prevRound => prevRound + 1)
        setAggregate(true)
      })

      socketServer.on('AddClientServer',(message)=>{
        setClientTraining(message)
      })

      socketServer.on('endAggregate' ,() => {
         
        setAggregate(false)
      })

      socketServer.on('downloadModel' ,(message) => {
        setListModels(message)
        setDownload(true)
      })

      socketServer.on('endDownloadModel' ,() => {
        setDownload(false)
      })



      setButtonServer(true);

     
    };

  

    const [selectedOptions, setSelectedOptions] = useState([]);

   


    
  // Funzione per gestire il cambiamento di stato delle checkbox
  const handleCheckboxChange = (option) => {
    if (selectedOptions.includes(option)) {
      setSelectedOptions(selectedOptions.filter((item) => item !== option));
    } else {
      setSelectedOptions([...selectedOptions, option]);
    }
    console.log(selectedOptions)
  };
    

  const handleButtonServer = () => {
    console.log(selectedOptions)
    const socketServer = io(`http://127.0.0.1:6020`);
    socketServer.emit('aggregate_fit_input', selectedOptions); // Invia il testo al server
    setSelectedOptions(''); // Azzera la casella di input dopo l'invio
  
};
    
  const handleCheckboxChangeDownload = (id) => {
    if (selectedId===id){
      setSelectedId(null)
    }
    else{
      setSelectedId(id);
    }
    
  };

  const handleSendToServerDownload = () => {
    const socketServer = io(`http://127.0.0.1:6020`);
    if (selectedId !== null) {
    
      socketServer.emit('downloadModel', selectedId);
      
  }
  else{socketServer.emit('downloadModel', ""); };
}

const openActionsPage = () => {
  window.open('/actions', '_blank');
};




    return (
      <><div className="container">
        <div className="button-container">
        {!buttonServer && (<button onClick={startserver} className='learning-button'>Start server</button>)}
          {buttonServer && (
            <div className="server-results">
              <h2>Server ON. Results:</h2>
              {serverResults.split('\n').map((line, index) => (
                <div key={index}><h3>{line}</h3></div>
              ))}

              
              <button className='showActions' onClick={openActionsPage}>Show all actions</button>
            </div>
          )}
          {!buttonClicked && (
            <button className="learning-button" onClick={startSimulation}>
              Start learning
            </button>
          )}
          


          <div>
            {serverAggregate && (

              <div className="server-form-container">

              <div className='checkbox'>
                    <h2 className='exclude-client'>Exclude clients:</h2>
                    {clientTraining.map((option, index) => (
                      <div key={index} className="checkbox-container">
                        <input
                          type="checkbox"
                          value={option}
                          checked={selectedOptions.includes(option)}
                          onChange={() => handleCheckboxChange(option)}
                        />
                        <label>Client {parseInt(option)+1}</label>
                      </div>
                    ))}
                    
                </div>
                    

                <button onClick={handleButtonServer} className="server-button">
                  Send to server
                </button>

              </div>)}


            {serverDownlaod && (
              <div className="Download">
               <div className='messageDownload'> <h3> load an already created model</h3></div>
              <div className='checkboxDownload'>
              {listModels.map(([name, date,id, accuracy ], index) => (
                <div key={index} className="tuple">
                  <input 
                    type="checkbox" 
                    value={id} 
                    checked={selectedId === id}
                    onChange={() => handleCheckboxChangeDownload(id)}
                  />
                  {`Name: ${name}, Accuracy: ${accuracy}, Date: ${date}`}
                </div>
              ))}
              </div>

              <button onClick={handleSendToServerDownload}>Send to server</button>
            </div>
            )}


          </div>
        </div>

        {buttonClicked && (
          <div className="dots-container">
            {dots.map(({ color, info }, index) => (
              <Dot
                key={index}
                id={index}
                color={color}
                info={info} />
            ))}
          </div>)}
      </div>
      

      

</>
  
  );



  
    
}

export default App;