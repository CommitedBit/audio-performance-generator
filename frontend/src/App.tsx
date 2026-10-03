import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import EditorPage from './pages/EditorPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  return (
    <BrowserRouter>
      <header className="appbar">
        <strong>Audio Performance Generator</strong>
        <nav>
          {/* The /settings route previously existed with nothing linking to it. */}
          <Link to="/">Editor</Link>
          <Link to="/settings">Server</Link>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/" element={<EditorPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
