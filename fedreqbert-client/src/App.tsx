import './App.css'
import Chat from "./components/Chat";

const App: React.FC = () => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-[#1a1b1c] px-4">
      <Chat />
    </div>
  )
}

export default App