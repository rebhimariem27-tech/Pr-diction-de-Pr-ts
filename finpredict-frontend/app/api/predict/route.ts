import { NextRequest, NextResponse } from 'next/server';

// Proxy serveur : la clé d'API reste côté serveur Next.js (jamais envoyée au navigateur).
// Variables SERVEUR (sans préfixe NEXT_PUBLIC_) : FINPREDICT_API_URL, FINPREDICT_API_KEY.
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const apiUrl = (process.env.FINPREDICT_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const apiKey = process.env.FINPREDICT_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ detail: 'FINPREDICT_API_KEY n’est pas configurée côté serveur Next.js.' }, { status: 500 });
  }

  let body: unknown;
  try { body = await req.json(); } catch { return NextResponse.json({ detail: 'Corps JSON invalide.' }, { status: 400 }); }

  try {
    const upstream = await fetch(`${apiUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json', 'X-API-Key': apiKey },
      body: JSON.stringify(body),
      cache: 'no-store',
      signal: AbortSignal.timeout(30_000),
    });
    const data = await upstream.json().catch(() => ({}));
    return NextResponse.json(data, { status: upstream.status });
  } catch {
    return NextResponse.json({ detail: `Impossible de joindre l’API à ${apiUrl}.` }, { status: 502 });
  }
}
