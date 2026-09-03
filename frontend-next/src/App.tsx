import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layouts/AppShell";
import { ComparePage } from "./pages/ComparePage";
import { CultureFitPage } from "./pages/CultureFitPage";
import { DemoPage } from "./pages/DemoPage";
import { LandingPage } from "./pages/LandingPage";
import { LoginPage } from "./pages/LoginPage";
import { MyPage } from "./pages/MyPage";
import { ReportLoadingPage } from "./pages/ReportLoadingPage";
import { ReportPage } from "./pages/ReportPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/demo" element={<DemoPage stage="self" />} />
          <Route path="/demo/interview" element={<DemoPage stage="interview" />} />
          <Route path="/demo/report/loading" element={<ReportLoadingPage />} />
          <Route path="/demo/report" element={<ReportPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/culture-fit" element={<CultureFitPage />} />
          <Route path="/my" element={<MyPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
