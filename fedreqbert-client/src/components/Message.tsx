// fedreqbert-client/src/components/Message.tsx
import React from "react";
import { type PredictionResponse } from "../context/AppContext";
import { FaCloud, FaBolt } from "react-icons/fa";

type MessageProps = {
  data: string | PredictionResponse;
  sender: "user" | "bot";
};

// Mapa de códigos a Nombres Completos y Colores
const getClassificationInfo = (code: string) => {
  const map: Record<string, { label: string; color: string }> = {
    // Binary
    'F':  { label: 'Funcional', color: 'text-blue-400' },
    'NF': { label: 'No Funcional', color: 'text-emerald-400' },
    
    // Multiclass (PROMISE NFR Dataset codes)
    'A':  { label: 'Availability', color: 'text-yellow-400' },
    'L':  { label: 'Legality', color: 'text-gray-400' },
    'LF': { label: 'Look & Feel', color: 'text-pink-400' },
    'MN': { label: 'Maintainability', color: 'text-teal-400' },
    'O':  { label: 'Operability', color: 'text-orange-400' },
    'PE': { label: 'Performance', color: 'text-amber-400' },
    'PO': { label: 'Portability', color: 'text-indigo-400' },
    'SC': { label: 'Scalability', color: 'text-cyan-400' },
    'SE': { label: 'Security', color: 'text-red-400' },
    'US': { label: 'Usability', color: 'text-lime-400' },
    'FT': { label: 'Fault Tolerance', color: 'text-rose-400' },
  };

  return map[code] || { label: 'Desconocido', color: 'text-gray-300' };
};

const Message: React.FC<MessageProps> = ({ data, sender }) => {
  
  // Renderizado simple para el usuario
  if (sender === "user") {
    return (
      <div className="flex justify-end w-full mb-4">
        <div className="max-w-[80%] p-4 rounded-2xl rounded-tr-none bg-[#343638] text-white shadow-md wrap-break-word">
          {typeof data === 'string' ? data : JSON.stringify(data)}
        </div>
      </div>
    );
  }

  // Renderizado para el BOT
  const isPrediction = typeof data !== 'string';
  
  // Obtenemos la info visual basada en la predicción (si es objeto)
  const classInfo = isPrediction 
    ? getClassificationInfo((data as PredictionResponse).prediction) 
    : { label: '', color: '' };

  return (
    <div className="flex justify-start w-full mb-4">
      <div className={`max-w-[90%] sm:max-w-[85%] p-0 rounded-2xl rounded-tl-none shadow-md overflow-hidden ${
        isPrediction ? "bg-[#1e1e1e]" : "bg-gray-200 text-black p-4"
      }`}>
        
        {!isPrediction ? (
          // Mensaje de texto simple
          <span>{data}</span>
        ) : (
          // TARJETA DE RESULTADOS
          <div className="flex flex-col text-white w-full min-w-75">
            
            {/* Cabecera de Clasificación Dinámica */}
            <div className="p-4 border-b border-gray-700 bg-[#252525]">
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Clasificación</p>
              <div className="flex items-center gap-3 flex-wrap">
                <span className={`text-2xl font-bold ${classInfo.color}`}>
                  {classInfo.label}
                </span>
                <span className="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-300 border border-gray-600">
                  {(data as PredictionResponse).prediction}
                </span>
              </div>
            </div>

            {/* Detalles de Infraestructura */}
            <div className="p-4 bg-[#1e1e1e] space-y-3">
              
              {/* Fuente (Source) */}
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Procesado en:</span>
                <div className={`flex items-center gap-2 px-2 py-1 rounded text-xs font-bold border ${
                  (data as PredictionResponse).offloaded 
                    ? 'bg-blue-900/20 border-blue-800 text-blue-300' // Cloud
                    : 'bg-green-900/20 border-green-800 text-green-300' // Fog
                }`}>
                  {(data as PredictionResponse).offloaded ? <FaCloud /> : <FaBolt />}
                  {(data as PredictionResponse).source}
                </div>
              </div>

              {/* Confianza */}
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Confianza:</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${(data as PredictionResponse).confidence > 0.8 ? 'bg-violet-500' : 'bg-yellow-500'}`}
                      style={{ width: `${(data as PredictionResponse).confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-mono text-white">
                    {((data as PredictionResponse).confidence * 100).toFixed(2)}%
                  </span>
                </div>
              </div>

            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Message;