const fs = require('fs');
const path = require('path');

// Get today's date in YYYY-MM-DD format
const today = new Date();
const year = today.getFullYear();
const month = String(today.getMonth() + 1).padStart(2, '0');
const day = String(today.getDate()).padStart(2, '0');
const dateStr = `${year}-${month}-${day}`;

const csvPath = path.join('register', `${dateStr}.csv`);

// Check if file already exists – if so, exit (do nothing)
if (fs.existsSync(csvPath)) {
  console.log(`File ${csvPath} already exists. Skipping creation.`);
  process.exit(0);
}

// Read students.json
let students = [];
try {
  const studentsFile = fs.readFileSync('students.json', 'utf8');
  const data = JSON.parse(studentsFile);
  // Assuming each student has a 'name' field
  students = data.map(s => s.name).filter(name => name);
} catch (err) {
  console.error('Failed to read students.json:', err.message);
  process.exit(1);
}

if (students.length === 0) {
  console.warn('No students found in students.json – creating empty file.');
}

// Build CSV content: each line is "name, Absent"
const csvContent = students.map(name => `${name},Absent`).join('\n');

// Ensure register folder exists
if (!fs.existsSync('register')) {
  fs.mkdirSync('register', { recursive: true });
}

// Write the file
fs.writeFileSync(csvPath, csvContent, 'utf8');
console.log(`Created ${csvPath} with ${students.length} students.`);
