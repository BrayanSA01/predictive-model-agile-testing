const fs = require('fs');

const totalRecords = 30000;

const testTypes = ['Funcional', 'No funcional'];
const methodologies = ['Ágil', 'Cascada'];
const testPhases = ['Estática', 'Dinámica'];
const testOrigins = ['Manual', 'Automatizada'];
const complexityLevels = ['Baja', 'Media', 'Alta', 'Crítica'];
const priorities = ['Baja', 'Media', 'Alta', 'Crítica'];

function randomFromArray(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function calculateFailureProbability(origin, complexity, priority) {
  let probability = 0.2;

  if (origin === 'Manual') probability += 0.15;  
  if (complexity === 'Alta') probability += 0.2;
  if (complexity === 'Crítica') probability += 0.3;
  if (priority === 'Crítica') probability += 0.1;

  return probability > 0.9 ? 0.9 : probability;
}

const records = [];

for (let i = 1; i <= totalRecords; i++) {
  const testType = randomFromArray(testTypes);
  const methodology = randomFromArray(methodologies);
  const testPhase = randomFromArray(testPhases);
  const testOrigin = randomFromArray(testOrigins);
  const complexity = randomFromArray(complexityLevels);
  const priority = randomFromArray(priorities);

  const failureProb = calculateFailureProbability(testOrigin, complexity, priority);
  const result = Math.random() < failureProb ? 'Fallida' : 'Aprobada';

  const record = {
    id: `TC${String(i).padStart(6, '0')}`,
    tipoPrueba: testType,
    metodologia: methodology,
    fase: testPhase,
    origen: testOrigin,
    complejidad: complexity,
    prioridad: priority,
    resultado: result
  };

  records.push(record);
}

const outputFileName = 'dataset_simplificado.json';
fs.writeFileSync(outputFileName, JSON.stringify(records, null, 2), 'utf-8');

console.log(`✅ Dataset generado en "${outputFileName}" con ${totalRecords} registros.`);
