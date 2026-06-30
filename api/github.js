// api/github.js
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const token = process.env.GH_TOKEN;
  const owner = process.env.GH_OWNER;
  const repo = process.env.GH_REPO;
  const branch = process.env.GH_BRANCH || 'main';

  if (!token || !owner || !repo) {
    console.error('Missing env vars');
    return res.status(500).json({ error: 'Missing GitHub env vars' });
  }

  const { path, method = 'GET', data = null } = req.body;
  if (!path) {
    return res.status(400).json({ error: 'Missing path' });
  }

  const baseUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;

  try {
    if (method === 'GET') {
      // Attempt to fetch the file
      let response = await fetch(`${baseUrl}?ref=${branch}`, {
        headers: {
          'Authorization': `token ${token}`,
          'Accept': 'application/vnd.github.v3+json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        return res.status(200).json(data);
      }

      // If 404, create the file with empty content
      if (response.status === 404) {
        console.log(`File not found, creating: ${path}`);
        const createResponse = await fetch(`${baseUrl}?ref=${branch}`, {
          method: 'PUT',
          headers: {
            'Authorization': `token ${token}`,
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: `Auto‑create ${path}`,
            content: Buffer.from('').toString('base64'), // empty file
            branch,
          }),
        });

        if (!createResponse.ok) {
          const err = await createResponse.json();
          console.error('Create failed:', err);
          return res.status(createResponse.status).json({ error: 'Create failed', details: err });
        }

        const created = await createResponse.json();
        // Return the created file's content (with sha)
        return res.status(200).json(created.content);
      }

      // Other errors
      const err = await response.json();
      return res.status(response.status).json({ error: err.message });
    }

    // PUT (or other methods)
    const options = {
      method,
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
      },
    };

    if (data) {
      // Remove sha if it's null/undefined (so GitHub creates a new file)
      if (data.sha === undefined || data.sha === null) {
        delete data.sha;
      }
      options.body = JSON.stringify(data);
    }

    console.log(`PUT request to ${path} with body:`, options.body);

    const response = await fetch(`${baseUrl}?ref=${branch}`, options);
    const responseData = await response.json();

    if (!response.ok) {
      console.error('GitHub PUT error:', response.status, responseData);
      return res.status(response.status).json({
        error: responseData.message || 'GitHub API error',
        details: responseData,
      });
    }

    return res.status(200).json(responseData);
  } catch (error) {
    console.error('Internal error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
