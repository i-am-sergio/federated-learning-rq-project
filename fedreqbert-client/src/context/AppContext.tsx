import React, { createContext, useState, useContext } from 'react';
import axios from 'axios';

// Definimos la estructura exacta de tu respuesta API
export interface PredictionResponse {
  prediction: string;
  confidence: number;
  source: string;
  offloaded: boolean;
}

interface AppContextProps {
  message: string;
  setMessage: React.Dispatch<React.SetStateAction<string>>;
  selectedModel: string;
  setSelectedModel: React.Dispatch<React.SetStateAction<string>>;
  // sendData ahora devuelve la estructura completa o null
  sendData: (textToSend: string) => Promise<PredictionResponse | null>;
}

const AppContext = createContext<AppContextProps | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [message, setMessage] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('Binary Model');

  // La URL de tu función Fog
  const API_URL = 'https://fog-node-fn-c159cd6-y5dphoazqq-uc.a.run.app';

  const sendData = async (textToSend: string): Promise<PredictionResponse | null> => {
    try {
      // Petición POST con la estructura {"text": "..."}
      const response = await axios.post(API_URL, {
        text: textToSend,
        // model: selectedModel // Podrías enviar esto si tu backend lo soporta en el futuro
      });

      console.log('Response from server:', response.data);
      
      // Retornamos los datos tal cual vienen del backend
      return response.data as PredictionResponse;
      
    } catch (error) {
      console.error('Error sending data:', error);
      return null;
    }
  };

  const value: AppContextProps = {
    message,
    setMessage,
    selectedModel,
    setSelectedModel,
    sendData,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};