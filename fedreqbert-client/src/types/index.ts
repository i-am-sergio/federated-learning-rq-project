// Estructura de la respuesta del servidor Flask
export interface PredictionResult {
  requirement: string;
  prediction: string;
  class_id: number;
  confidence: number;
}

// Estructura de un mensaje en el chat
export interface Message {
  id: string;
  role: 'user' | 'bot';
  content?: string; // Para mensajes simples o errores
  predictionData?: PredictionResult | null; // Para respuestas exitosas del bot
  isLoading?: boolean; // Para mostrar el estado de "pensando"
  isError?: boolean;
}

// Definición de modelos disponibles
export interface ModelOption {
  name: string;
  url: string;
}


// Reemplaza la IP con la de tu servidor real
export const AVAILABLE_MODELS: ModelOption[] = [
  { name: "FedReqBERT MPNet (Binario)", url: "http://34.9.5.148:5000/predict" },
  { name: "FedReqBERT MpNet (Multiclase)", url: "http://34.9.5.148:5000/predict_multiclass" },
];