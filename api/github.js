// api/github.js
export default async function handler(req, res) {
  // Only POST requests are accepted
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const token = process.env.GH_TOKEN;
  const owner = process.env.GH_OWNER;
  const repo = process.env.GH_REPO;

  if (!token || !owner || !repo) {
    return res.status(500).json({ error: 'Missing GitHub environment variables' });
  }

  const { path, method = 'GET', data = null } = req.body;

  if (!path) {
    return res.status(400).json({ error: 'Missing path' });
  }

  const url = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;

  const options = {
    method,
    headers: {
      'Authorization': `token ${token}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
    },
  };

  if (data) {
    options.body = JSON.stringify(data);
  }

  try {
    const response = await fetch(url, options);
    const responseData = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({ error: responseData.message || 'GitHub API error' });
    }

    return res.status(200).json(responseData);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
