import React, { useState, useRef, useEffect } from "react";
import Message from "./Message";
import ChatInput from "./ChatInput";
import Navbar from "./Navbar";
import { useAppContext, type PredictionResponse } from "../context/AppContext";

type MessageState = {
  data: string | PredictionResponse;
  sender: "user" | "bot";
};

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<MessageState[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { setMessage, sendData } = useAppContext();

  const handleSendMessage = async (textInput: string) => {
    setMessages((prev) => [...prev, { data: textInput, sender: "user" }]);
    setMessage(textInput); 
    setMessages((prev) => [...prev, { data: "Analizando requisito...", sender: "bot" }]);

    try {
      const response = await sendData(textInput);
      setMessages((prev) => prev.filter((msg) => msg.data !== "Analizando requisito..."));

      if (response) {
        setMessages((prev) => [...prev, { data: response, sender: "bot" }]);
      } else {
        setMessages((prev) => [
          ...prev,
          { data: "Lo siento, hubo un error al procesar tu solicitud.", sender: "bot" },
        ]);
      }
    } catch (error) {
      console.error("Error al obtener la respuesta:", error);
      setMessages((prev) => prev.filter((msg) => msg.data !== "Analizando requisito..."));
      setMessages((prev) => [
        ...prev,
        { data: "Error crítico de conexión.", sender: "bot" },
      ]);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex flex-col w-full h-screen bg-[#1c1c1c]">
      
      {/* Header Fijo */}
      <div className="w-full border-b border-gray-800 shrink-0">
        <div className="max-w-4xl mx-auto px-4">
          <Navbar />
        </div>
      </div>
      
      {/* Área de Mensajes (Flexible) 
         AQUÍ ES DONDE FALTABAN LAS CLASES DEL SCROLLBAR
      */}
      <div className="flex-1 w-full relative overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent hover:scrollbar-thumb-gray-500">
        
        {/* Contenedor centrado */}
        <div className="max-w-4xl mx-auto w-full h-full">
          <div className="flex flex-col px-4 py-4 space-y-2 min-h-full"> 
            
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center flex-1 opacity-80">
                <div className="text-center text-gray-400 text-3xl font-light mb-4">
                  FedReq<span className="font-bold text-violet-400">BERT</span>
                </div>
                <p className="text-gray-500 max-w-md text-center">
                  Clasificación híbrida (Fog/Cloud) de requisitos de software utilizando aprendizaje federado.
                </p>
              </div>
            ) : (
              <>
                {messages.map((msg, index) => (
                  <Message key={index} data={msg.data} sender={msg.sender} />
                ))}
                <div ref={messagesEndRef} className="h-2" />
              </>
            )}
          </div>
        </div>
      </div>

      {/* Área del Input (Fija abajo) */}
      <div className="w-full bg-[#1c1c1c] shrink-0 border-t border-gray-800">
        <div className="max-w-4xl mx-auto px-4 pb-6 pt-2">
          <ChatInput onSend={handleSendMessage} />
        </div>
      </div>
    </div>
  );
};

export default Chat;