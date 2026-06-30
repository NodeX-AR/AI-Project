// api/github.js
export default async function handler(req, res) {
  // Only POST requests are accepted
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const token = process.env.GH_TOKEN;
  const owner = process.env.GH_OWNER;
  const repo = process.env.GH_REPO;
  const branch = process.env.GH_BRANCH || 'main';

  if (!token || !owner || !repo) {
    console.error('Missing env vars:', { token: !!token, owner: !!owner, repo: !!repo });
    return res.status(500).json({ error: 'Missing GitHub environment variables' });
  }

  const { path, method = 'GET', data = null } = req.body;

  if (!path) {
    return res.status(400).json({ error: 'Missing path' });
  }

  // Build the GitHub API URL
  const baseUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;

  try {
    if (method === 'GET') {
      // Try to fetch the file
      let response = await fetch(`${baseUrl}?ref=${branch}`, {
        headers: {
          'Authorization': `token ${token}`,
          'Accept': 'application/vnd.github.v3+json',
        },
      });

      // If file exists, return it
      if (response.ok) {
        const data = await response.json();
        return res.status(200).json(data);
      }

      // If file does NOT exist (404), create it with default content
      if (response.status === 404) {
        console.log(`File ${path} not found – creating default.`);

        // Default content: an empty CSV (just a header or nothing)
        // You can change this to any default, e.g., "Name,Status\n" or ""
        const defaultContent = '';  // empty file – frontend will treat as no records

        const createResponse = await fetch(`${baseUrl}?ref=${branch}`, {
          method: 'PUT',
          headers: {
            'Authorization': `token ${token}`,
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: `Create missing file: ${path}`,
            content: Buffer.from(defaultContent).toString('base64'),
            branch,
          }),
        });

        if (!createResponse.ok) {
          const errorData = await createResponse.json();
          console.error('Failed to create file:', errorData);
          return res.status(createResponse.status).json({
            error: 'Failed to create missing file',
            details: errorData,
          });
        }

        // Return the newly created file's content (same as if it existed)
        const createdData = await createResponse.json();
        return res.status(200).json(createdData.content);
      }

      // Other errors (e.g., 401, 403) – pass through
      const errorData = await response.json();
      return res.status(response.status).json({ error: errorData.message || 'GitHub API error' });
    }

    // For PUT (and other methods), proxy as before
    const options = {
      method,
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
      },
    };

    if (data) {
      // If 'sha' is undefined or null, remove it so GitHub creates a new file
      if (data.sha === undefined || data.sha === null) {
        delete data.sha;
      }
      options.body = JSON.stringify(data);
    }

    const response = await fetch(`${baseUrl}?ref=${branch}`, options);
    const responseData = await response.json();

    if (!response.ok) {
      console.error('GitHub API error:', response.status, responseData);
      return res.status(response.status).json({ error: responseData.message || 'GitHub API error' });
    }

    return res.status(200).json(responseData);
  } catch (error) {
    console.error('Internal error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
