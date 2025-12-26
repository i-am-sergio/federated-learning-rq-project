import React, { useState } from "react";
import { IoMdSend } from "react-icons/io";
import { GoPaperclip } from "react-icons/go";

type ChatInputProps = {
  onSend: (message: string) => void;
};

const ChatInput: React.FC<ChatInputProps> = ({ onSend }) => {
  const [inputValue, setInputValue] = useState("");

  const handleSend = () => {
    if (inputValue.trim()) {
      onSend(inputValue); 
      setInputValue("");
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" && inputValue.trim()) {
      event.preventDefault();
      handleSend();
    }
  };
  
  return (
    <div className="p-0 bg-transparent">
      <div className="flex items-center p-2 bg-[#27292b] border border-gray-600 rounded-full shadow-lg">
        
        {/* Botón Adjuntar (Placeholder) */}
        <button
          className="p-2 text-gray-400 hover:text-white rounded-full transition-colors mx-1"
          onClick={() => console.log("Adjuntar no implementado")}
          title="Adjuntar archivo de requisitos"
        >
          <GoPaperclip className="w-6 h-6" />
        </button>

        <input
          type="text"
          className="flex-1 bg-transparent text-white text-lg px-4 focus:outline-none placeholder-gray-500"
          placeholder="Escribe un requisito (ej: The system must...)"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
        />

        <button
          className={`p-3 rounded-full transition-all duration-200 flex items-center justify-center ${
            inputValue.trim()
              ? "bg-violet-600 text-white hover:bg-violet-700 shadow-violet-500/20 shadow-lg"
              : "bg-gray-700 text-gray-500 cursor-not-allowed"
          }`}
          onClick={handleSend}
          disabled={!inputValue.trim()}
        >
          <IoMdSend className="w-5 h-5 ml-0.5" /> {/* Ajuste visual leve del icono */}
        </button>
      </div>
    </div>
  );
};

export default ChatInput;