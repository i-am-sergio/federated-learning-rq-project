import React from 'react';
import { AVAILABLE_MODELS, type ModelOption } from '../types';

interface HeaderProps {
  currentModelUrl: string;
  onModelChange: (url: string) => void;
  disabled: boolean;
}

export const Header: React.FC<HeaderProps> = ({ currentModelUrl, onModelChange, disabled }) => {
  return (
    <header className="fixed top-0 left-0 right-0 bg-gray-900/90 backdrop-blur-sm border-b border-white/10 p-3 z-50">
      <div className="max-w-3xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2">
           <span className="text-gray-200 font-semibold tracking-tight">FedReqBERT</span>
           <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-500/20 text-blue-300 font-medium border border-blue-500/30">
             BETA
           </span>
        </div>
        
        <div className="relative group">
          <select
            value={currentModelUrl}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={disabled}
            className="appearance-none bg-gray-800 text-sm text-gray-300 border border-gray-700 rounded-md py-1.5 pl-3 pr-8 focus:outline-none focus:ring-1 focus:ring-gray-500 cursor-pointer hover:bg-gray-700 transition-colors"
          >
            {AVAILABLE_MODELS.map((model: ModelOption) => (
              <option key={model.url} value={model.url}>
                {model.name}
              </option>
            ))}
          </select>
          {/* Flecha personalizada del select */}
          <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
            <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
          </div>
        </div>
      </div>
    </header>
  );
};