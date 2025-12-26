// fedreqbert-client/src/App.tsx
import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Header } from './components/Header';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { type Message, type PredictionResult, AVAILABLE_MODELS } from './types';

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uuidv4(),
      role: 'bot',
      content: "Hola. Soy FedReqBERT, un modelo de inteligencia artificial federado.\n\nPuedo ayudarte a clasificar requisitos de software en Funcionales (F) o No Funcionales (NF). ¿Cuál es el requisito que deseas analizar hoy?"
    }
  ]);
  const [currentModelUrl, setCurrentModelUrl] = useState<string>(AVAILABLE_MODELS[0].url);
  const [loading, setLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (text: string) => {
    const userMessage: Message = { id: uuidv4(), role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    const botLoadingId = uuidv4();
    setMessages(prev => [...prev, { id: botLoadingId, role: 'bot', isLoading: true }]);

    try {
      const response = await fetch(currentModelUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text }),
      });

      if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);

      const data: PredictionResult = await response.json();

      setMessages(prev => prev.map(msg => 
        msg.id === botLoadingId 
          ? { ...msg, isLoading: false, predictionData: data }
          : msg
      ));

    } catch (err) {
      console.error(err);
      setMessages(prev => prev.map(msg => 
        msg.id === botLoadingId 
          ? { ...msg, isLoading: false, isError: true, content: "Error de conexión con el servidor Serverless." }
          : msg
      ));
    } finally {
      setLoading(false);
    }
  };

  return (
    // Fondo general oscuro (estilo ChatGPT Dark Mode)
    <div className="flex flex-col h-screen bg-[#343541] text-gray-100 font-sans">
      
      <Header 
        currentModelUrl={currentModelUrl}
        onModelChange={setCurrentModelUrl}
        disabled={loading}
      />

      {/* Contenedor Scrollable */}
      <main className="flex-1 overflow-y-auto w-full scroll-smooth pt-16 pb-36">
        <div className="flex flex-col w-full">
           {/* Si no hay mensajes (solo bienvenida), podríamos mostrar sugerencias aquí */}
           
           {messages.map(message => (
             <ChatMessage key={message.id} message={message} />
           ))}
           
           <div ref={messagesEndRef} className="h-4" />
        </div>
      </main>

      <ChatInput onSend={handleSendMessage} disabled={loading} />
      
    </div>
  );
}

export default App;