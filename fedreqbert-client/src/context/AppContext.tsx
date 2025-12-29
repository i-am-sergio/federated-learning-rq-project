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
  sendData: (textToSend: string) => Promise<PredictionResponse | null>;
}

const AppContext = createContext<AppContextProps | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [message, setMessage] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('Binary Model');

  // URL del Fog Node (La misma para ambos casos según tu curl)
  const API_URL = 'https://fog-node-fn-c159cd6-y5dphoazqq-uc.a.run.app';

  const sendData = async (textToSend: string): Promise<PredictionResponse | null> => {
    
    // Construcción del Payload dinámico
    const payload: { text: string; mode?: string } = {
      text: textToSend
    };

    // Si el modelo es Multiclass, agregamos el campo 'mode'
    if (selectedModel === 'Multiclass Model') {
      payload.mode = 'multiclass';
    }

    console.log(`Sending request to [${selectedModel}] at ${API_URL} with payload:`, payload);

    try {
      const response = await axios.post(API_URL, payload);

      console.log('Response from server:', response.data);
      return response.data as PredictionResponse;
      
    } catch (error) {
      console.error(`Error sending data to ${selectedModel}:`, error);
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