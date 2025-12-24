import React from 'react';
import { type Message } from '../types';
import { BotIcon, UserIcon } from './ChatIcons';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const isError = message.isError;

  return (
    <div className={`w-full border-b border-black/5 dark:border-white/5 ${isUser ? 'bg-[#343541]/0' : 'bg-[#444654]/40'}`}>
      <div className="max-w-3xl mx-auto flex gap-4 p-4 md:p-6 text-base md:gap-6">
        
        {/* Columna Avatar */}
        <div className="flex-shrink-0 flex flex-col relative items-end">
          {isUser ? <UserIcon /> : <BotIcon />}
        </div>

        {/* Columna Contenido */}
        <div className="relative flex-1 overflow-hidden">
          
          {/* Nombre (Opcional, estilo chat grupal) o solo contenido */}
          <div className="font-semibold text-gray-100 mb-1 opacity-90">
             {isUser ? 'Tú' : 'FedReqBERT'}
          </div>

          {/* Loading State */}
          {message.isLoading && (
            <div className="flex items-center space-x-1 h-6">
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></span>
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-75"></span>
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-150"></span>
            </div>
          )}

          {/* Texto Simple */}
          {message.content && (
             <div className={`whitespace-pre-wrap leading-7 text-gray-300 ${isError ? 'text-red-400' : ''}`}>
               {message.content}
             </div>
          )}

          {/* Tarjeta de Predicción (Solo Bot) */}
          {!isUser && message.predictionData && (
            <div className="mt-4 bg-gray-900/50 border border-gray-700 rounded-md overflow-hidden max-w-lg">
              <div className="bg-gray-800/80 px-4 py-2 border-b border-gray-700 flex justify-between items-center">
                <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Resultado del Análisis</span>
                <span className="text-xs text-gray-500 font-mono">ID: {message.predictionData.class_id}</span>
              </div>
              <div className="p-4">
                 <p className="text-sm text-gray-400 mb-3 italic">"{message.predictionData.requirement}"</p>
                 
                 <div className="flex items-center gap-4">
                    <div className="flex flex-col">
                       <span className="text-xs text-gray-500 mb-1">Clasificación</span>
                       <span className={`inline-flex px-3 py-1 rounded-full text-sm font-semibold
                          ${message.predictionData.prediction.includes('No Funcional') 
                             ? 'bg-purple-900/30 text-purple-300 border border-purple-700/50' 
                             : 'bg-emerald-900/30 text-emerald-300 border border-emerald-700/50'
                          }`}>
                          {message.predictionData.prediction}
                       </span>
                    </div>

                    <div className="w-px h-8 bg-gray-700"></div>

                    <div className="flex flex-col">
                       <span className="text-xs text-gray-500 mb-1">Confianza</span>
                       <span className="text-yellow-400 font-mono font-medium">
                          {(message.predictionData.confidence * 100).toFixed(2)}%
                       </span>
                    </div>
                 </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};