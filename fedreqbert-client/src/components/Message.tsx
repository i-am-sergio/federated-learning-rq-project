import React from "react";
import { type PredictionResponse } from "../context/AppContext";
import { FaCloud, FaBolt } from "react-icons/fa";

type MessageProps = {
  data: string | PredictionResponse; // Puede ser texto simple o objeto de predicción
  sender: "user" | "bot";
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

  // Renderizado para el BOT (Puede ser texto de error/pensando o la predicción real)
  const isPrediction = typeof data !== 'string';

  return (
    <div className="flex justify-start w-full mb-4">
      <div className={`max-w-[85%] p-0 rounded-2xl rounded-tl-none shadow-md overflow-hidden ${
        isPrediction ? "bg-[#1e1e1e]" : "bg-gray-200 text-black p-4"
      }`}>
        
        {!isPrediction ? (
          // Mensaje de texto simple (ej: "Pensando..." o Error)
          <span>{data}</span>
        ) : (
          // TARJETA DE RESULTADOS (Fog vs Cloud)
          <div className="flex flex-col text-white w-full min-w-75">
            
            {/* Cabecera de Clasificación */}
            <div className="p-4 border-b border-gray-700 bg-[#252525]">
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Clasificación</p>
              <div className="flex items-center gap-3">
                <span className={`text-2xl font-bold ${
                  (data as PredictionResponse).prediction === 'F' ? 'text-blue-400' : 'text-emerald-400'
                }`}>
                  {(data as PredictionResponse).prediction === 'F' ? 'Funcional' : 'No Funcional'}
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
                    ? 'bg-blue-900/20 border-blue-800 text-blue-300' // Estilo Cloud
                    : 'bg-green-900/20 border-green-800 text-green-300' // Estilo Fog
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