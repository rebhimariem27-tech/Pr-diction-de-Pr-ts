import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const apiUrl = (process.env.FINPREDICT_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const apiKey = process.env.FINPREDICT_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ detail: 'FINPREDICT_API_KEY n’est pas configurée côté serveur Next.js.' }, { status: 500 });
  }

  const query = req.nextUrl.searchParams.toString();
  try {
    const upstream = await fetch(`${apiUrl}/dashboard-summary${query ? `?${query}` : ''}`, {
      headers: { Accept: 'application/json', 'X-API-Key': apiKey },
      cache: 'no-store',
      signal: AbortSignal.timeout(30_000),
    });
    const data = await upstream.json().catch(() => ({}));
    return NextResponse.json(data, { status: upstream.status });
  } catch {
    return NextResponse.json({ detail: `Impossible de joindre l’API à ${apiUrl}.` }, { status: 502 });
  }
}
