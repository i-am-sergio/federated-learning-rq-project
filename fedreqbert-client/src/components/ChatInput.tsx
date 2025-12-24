import React, { useState, useRef, useEffect } from 'react';

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, disabled }) => {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input);
      setInput("");
      if (textareaRef.current) textareaRef.current.style.height = '24px'; // Reset height
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      // Limitamos la altura máxima
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 200)}px`;
    }
  }, [input]);

  return (
    <div className="fixed bottom-0 left-0 w-full bg-gradient-to-t from-gray-900 via-gray-900 to-transparent pb-6 pt-10 px-4 z-20">
      <div className="max-w-3xl mx-auto">
        {/* El contenedor maneja el borde visual al hacer foco (focus-within) */}
        <div className="relative flex items-end w-full p-3 bg-[#40414f] border border-gray-600/50 rounded-xl shadow-lg focus-within:border-gray-500/80 focus-within:ring-1 focus-within:ring-gray-500/80 transition-all">
          
          <textarea
            ref={textareaRef}
            // AQUÍ ESTÁ EL CAMBIO: agregué 'outline-none focus:outline-none' y aseguré 'focus:ring-0'
            className="w-full max-h-[200px] py-[2px] pr-10 pl-2 bg-transparent border-0 outline-none focus:outline-none focus:ring-0 text-gray-100 placeholder-gray-400 resize-none m-0 overflow-y-hidden"
            rows={1}
            placeholder="Escribe un requisito para clasificar..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            style={{ height: '24px' }} 
          />
          
          <button
            onClick={() => handleSubmit()}
            disabled={disabled || !input.trim()}
            className="absolute right-3 bottom-2.5 p-1.5 rounded-md text-gray-400 hover:bg-gray-900/50 hover:text-gray-200 disabled:opacity-40 disabled:hover:bg-transparent transition-all"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
              <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
            </svg>
          </button>
        </div>
        <div className="text-center mt-2">
            <p className="text-[10px] text-gray-500">FedReqBERT puede cometer errores. Considera verificar la información importante.</p>
        </div>
      </div>
    </div>
  );
};