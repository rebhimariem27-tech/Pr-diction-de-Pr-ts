import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ applicationId: string }> },
) {
  const apiUrl = (process.env.FINPREDICT_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const apiKey = process.env.FINPREDICT_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ detail: 'FINPREDICT_API_KEY n’est pas configurée côté serveur Next.js.' }, { status: 500 });
  }

  const { applicationId } = await params;
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: 'Corps JSON invalide.' }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${apiUrl}/applications/${encodeURIComponent(applicationId)}/decision`, {
      method: 'PATCH',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-API-Key': apiKey },
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
