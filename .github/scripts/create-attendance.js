const fs = require('fs');
const path = require('path');

const today = new Date();
const year = today.getFullYear();
const month = String(today.getMonth() + 1).padStart(2, '0');
const day = String(today.getDate()).padStart(2, '0');
const dateStr = `${year}-${month}-${day}`;

const csvPath = path.join('register', `${dateStr}.csv`);


if (fs.existsSync(csvPath)) {
  console.log(`File ${csvPath} already exists. Skipping creation.`);
  process.exit(0);
}


let students = [];
try {
  const studentsFile = fs.readFileSync('students.json', 'utf8');
  const data = JSON.parse(studentsFile);
  
  students = data.map(s => s.name).filter(name => name);
} catch (err) {
  console.error('Failed to read students.json:', err.message);
  process.exit(1);
}

if (students.length === 0) {
  console.warn('No students found in students.json – creating empty file.');
}


const csvContent = students.map(name => `${name},Absent`).join('\n');

if (!fs.existsSync('register')) {
  fs.mkdirSync('register', { recursive: true });
}


fs.writeFileSync(csvPath, csvContent, 'utf8');
console.log(`Created ${csvPath} with ${students.length} students.`);
