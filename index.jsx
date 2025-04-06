import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";

function MatchDetail({ matchId }) {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <h1 className="text-3xl font-bold mb-4">Chi tiết trận đấu #{matchId}</h1>
      <div className="aspect-video bg-black rounded-xl overflow-hidden mb-6">
        <iframe
          className="w-full h-full"
          src="https://www.youtube.com/embed/dQw4w9WgXcQ"
          title="Live Stream"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        ></iframe>
      </div>
      <p className="text-gray-300">Thông tin bình luận viên, đội hình, số liệu và nhận định trận đấu sẽ được hiển thị ở đây.</p>
    </div>
  </div>
  );
}

function HomePage() {
  const [search, setSearch] = useState("");
  const navigate = useNavigate();
  return {
return {
    <div className="min-h-screen bg-gray-900 text-white font-sans">
        <header className="bg-gray-800 shadow-md sticky top-0 z-50">
            <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                <h1 className="text-2xl font-bold text-pink-500 cursor-pointer" onClick={() => navigate("/")}>CK09.TV</h1>
                <nav className="space-x-4 text-sm md:text-base">
                    <a href="#" className="hover:text-pink-400">Trang chủ</a>
                    <a href="#" className="hover:text-pink-400">Lịch trực tiếp</a>
                    <a href="#" className="hover:text-pink-400">Khuyến mãi</a>
                    <a href="#" className="hover:text-pink-400">Tin tức</a>
                    <a href="#" className="hover:text-pink-400">Tải App</a>
                </nav>
            </div>
        </header>

        <div className="fixed bottom-4 right-4 bg-pink-600 text-white px-4 py-3 rounded-xl shadow-lg animate-bounce z-50">
            🎉 Khuyến mãi đặc biệt hôm nay! <a href="#" className="underline ml-2">Xem ngay</a>
        </div>

        <main className="container mx-auto px-4 py-10 flex flex-col lg:flex-row gap-8">
            <div className="w-full lg:w-3/4">
                <div className="mb-6">
            <input
                type="text"
                placeholder="Tìm kiếm"
                className="w-full p-3 rounded-xl bg-gray-800 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-pink-500"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <h2 className="text-3xl font-semibold mb-6">Trực tiếp hôm nay</h2>

          <div className="grid gap-6 md:grid-cols-2">
            {[1, 2, 3, 4, 5, 6].map((match) => (
              <div key={match} className="bg-gray-800 rounded-2xl shadow-lg p-4 hover:shadow-xl transition">
                <img
                  src="https://via.placeholder.com/400x200.png?text=Match+Thumbnail"
                  alt="Match Thumbnail"
                  className="rounded-xl mb-4 w-full h-40 object-cover"
                />
                <h3 className="text-xl font-bold mb-2">Trận đấu {match}</h3>
                <p className="text-sm text-gray-400 mb-4">19:00 | 06/04/2025</p>
                <button
                  className="bg-pink-600 hover:bg-pink-500 transition px-4 py-2 rounded-full text-white text-sm"
                  onClick={() => navigate(`/match/${match}`)}
                >
                  Xem ngay
                </button>
              </div>
            ))}
          </div>

          <div className="mt-12">
            <h3 className="text-2xl font-bold mb-4">Trận sắp tới</h3>
            <div className="overflow-x-auto flex gap-4 pb-2">
              {[1, 2, 3].map((id) => (
                <div key={id} className="min-w-[300px] bg-gray-800 rounded-xl p-4 shadow-md flex-shrink-0">
                  <img
                    src={`https://via.placeholder.com/300x150.png?text=Sắp+Tới+${id}`}
                    className="rounded mb-2"
                    alt="Next Match"
                  />
                  <p className="text-white font-semibold">Trận sắp tới #{id}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <aside className="w-full lg:w-1/4 space-y-6">
          <div className="bg-gray-800 p-4 rounded-xl shadow">
            <h4 className="text-lg font-bold mb-2">Tải App</h4>
            <p className="text-sm text-gray-400 mb-2">Xem mọi lúc mọi nơi</p>
            <button className="bg-pink-600 px-4 py-2 rounded-full text-sm">Tải ngay</button>
          </div>

          <div className="bg-gray-800 p-4 rounded-xl shadow">
            <h4 className="text-lg font-bold mb-2">BLV nổi bật</h4>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>🔥 Anh Tú</li>
              <li>🎙️ Minh Vũ</li>
              <li>⚽ Kiên Phạm</li>
            </ul>
          </div>

          <div className="bg-gray-800 p-4 rounded-xl shadow">
            <h4 className="text-lg font-bold mb-2">Quảng cáo</h4>
            <img
              src="https://via.placeholder.com/300x250.png?text=Ad"
              alt="Ad"
              className="rounded"
            />
          </div>
        </aside>
      </main>

      <footer className="bg-gray-800 mt-12 py-6 text-center text-gray-400 text-sm">
        © 2025 CK09.TV Clone - Phiên bản demo. Liên hệ: support@example.com
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/match/:matchId" element={<MatchDetail />} />
      </Routes>
    </Router>
  );
}
