import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'FinPredict AI Pro — Intelligent Credit Risk',
  description: 'Plateforme d’analyse du risque de crédit, explicable et suivie de bout en bout.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
