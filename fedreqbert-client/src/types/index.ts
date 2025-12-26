// fedreqbert-client/src/types/index.ts
// Estructura de la respuesta del servidor Flask
export interface PredictionResult {
  // Datos que vienen de tu API Python
  prediction: string;      // "F" o "NF"
  confidence: number;      // 0.99
  source: string;          // "FOG (FastModel)" o "CLOUD (DeepModel)"
  offloaded: boolean;      // true o false
  
  // Opcionales (por si quieres mantener compatibilidad con código viejo)
  requirement?: string;
  class_id?: number; 
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
  { name: "FedReqBERT MPNet (Binario)", url: "https://fog-node-fn-c159cd6-y5dphoazqq-uc.a.run.app" },
  { name: "FedReqBERT MpNet (Multiclase)", url: "http://34.9.5.148:5000/predict_multiclass" },
];