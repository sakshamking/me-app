// M+E Intelligence Engine V3 — Browser Edition
// Ported from pilot-v2/engine/ (Node.js) to client-side JavaScript
// All classes run in-browser. No server dependency.

// ============================================================
// CALIBRATOR — Baseline Learning (2 min window)
// ============================================================

class Calibrator {
  constructor(options = {}) {
    this.calibrationDuration = options.calibrationDuration || 120000;
    this.startTime = null;
    this.calibrated = false;
    this.samples = {
      engagement: [], eyeOpenRatio: [], headStability: [],
      movement: [], heartRate: [], browActivity: [], nodActivity: []
    };
    this.baseline = null;
  }

  start() {
    this.startTime = Date.now();
    this.calibrated = false;
    this.baseline = null;
    for (const key of Object.keys(this.samples)) this.samples[key] = [];
  }

  addFaceData(data) {
    if (this.calibrated) return;
    this.samples.engagement.push(data.engagement || 50);
    this.samples.eyeOpenRatio.push(data.eyes === 'Open' ? 1 : data.eyes === 'Droopy' ? 0.5 : 0);
    this.samples.headStability.push(data.headPose === 'Forward' ? 1 : data.headPose === 'Down' ? 0.3 : 0.7);
    this.samples.browActivity.push(data.brow === 'Raised' ? 1 : 0);
    this.samples.nodActivity.push(data.nod === 'Active' ? 1 : data.nod === 'Light' ? 0.5 : 0);
  }

  addSensorData(data) {
    if (this.calibrated) return;
    if (data.accel) {
      const mag = Math.sqrt(data.accel.x ** 2 + data.accel.y ** 2 + data.accel.z ** 2);
      this.samples.movement.push(Math.abs(mag - 9.8));
    }
    if (data.hr && data.hr > 0) this.samples.heartRate.push(data.hr);
  }

  tick() {
    if (this.calibrated) return true;
    if (!this.startTime) return false;
    if (Date.now() - this.startTime < this.calibrationDuration) return false;
    if (this.samples.engagement.length < 5 && this.samples.movement.length < 10) return false;
    this.baseline = this._computeBaseline();
    this.calibrated = true;
    return true;
  }

  _computeBaseline() {
    const avg = arr => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
    const std = arr => {
      if (arr.length < 2) return 0;
      const m = avg(arr);
      return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
    };
    return {
      engagement: { mean: avg(this.samples.engagement), std: std(this.samples.engagement) },
      eyeOpenRatio: { mean: avg(this.samples.eyeOpenRatio), std: std(this.samples.eyeOpenRatio) },
      headStability: { mean: avg(this.samples.headStability), std: std(this.samples.headStability) },
      movement: { mean: avg(this.samples.movement), std: std(this.samples.movement) },
      heartRate: this.samples.heartRate.length > 3
        ? { mean: avg(this.samples.heartRate), std: std(this.samples.heartRate) } : null,
      browActivity: { mean: avg(this.samples.browActivity), std: std(this.samples.browActivity) },
      nodActivity: { mean: avg(this.samples.nodActivity), std: std(this.samples.nodActivity) },
      sampleCount: Math.max(this.samples.engagement.length, this.samples.movement.length),
      calibratedAt: Date.now()
    };
  }

  getDelta(signal, currentValue) {
    if (!this.baseline || !this.baseline[signal]) return 0;
    const b = this.baseline[signal];
    if (b.std === 0 || b.mean === null) return 0;
    return (currentValue - b.mean) / Math.max(b.std, 0.01);
  }

  getProgress() {
    if (this.calibrated) return 1;
    if (!this.startTime) return 0;
    return Math.min(1, (Date.now() - this.startTime) / this.calibrationDuration);
  }

  toJSON() {
    return {
      calibrated: this.calibrated, progress: this.getProgress(),
      baseline: this.baseline,
      sampleCounts: Object.fromEntries(Object.entries(this.samples).map(([k, v]) => [k, v.length]))
    };
  }
}

// ============================================================
// STATE ESTIMATOR — Multi-signal fusion, hold-last-value, confidence decay
// ============================================================

class StateEstimator {
  constructor(calibrator) {
    this.calibrator = calibrator;
    this.faceHistory = [];
    this.sensorHistory = [];
    this.stateHistory = [];
    this.state = {
      energy: { level: 'medium', value: 0.5, confidence: 0 },
      immersion: { level: 'observing', value: 0.4, confidence: 0 },
      trajectory: { direction: 'flat', value: 0, confidence: 0 },
      raw: {}
    };
    this.lastFaceEnergy = null;
    this.lastFaceImmersion = null;
    this.FACE_STALE_MS = 30000;
    this.FACE_DECAY_MS = 120000;
    this.MAX_FACE_HISTORY = 12;
    this.MAX_SENSOR_HISTORY = 60;
    this.MAX_STATE_HISTORY = 60;
  }

  addFaceData(data) {
    this.faceHistory.push({ ...data, t: Date.now() });
    if (this.faceHistory.length > this.MAX_FACE_HISTORY) this.faceHistory.shift();
  }

  addSensorData(data) {
    this.sensorHistory.push({ ...data, t: Date.now() });
    if (this.sensorHistory.length > this.MAX_SENSOR_HISTORY) this.sensorHistory.shift();
  }

  estimate(context) {
    const energy = this._estimateEnergy(context);
    const immersion = this._estimateImmersion(context);
    this.stateHistory.push({ t: Date.now(), energy: energy.value, immersion: immersion.value });
    if (this.stateHistory.length > this.MAX_STATE_HISTORY) this.stateHistory.shift();
    const trajectory = this._estimateTrajectory();
    this.state = { energy, immersion, trajectory, raw: this._getRawSignals(), timestamp: Date.now() };
    return this.state;
  }

  _estimateEnergy(context) {
    const now = Date.now();
    const recent = this.faceHistory.slice(-6);
    const recentSensors = this.sensorHistory.slice(-10);
    const latestFaceTime = recent.length > 0 ? recent[recent.length - 1].t : 0;
    const faceAge = now - latestFaceTime;
    const faceFresh = recent.length > 0 && faceAge < this.FACE_STALE_MS;

    let faceValue = null, faceConfidence = 0;

    if (faceFresh) {
      let energyScore = 0, weights = 0;
      for (const f of recent) {
        let fe = 0.5;
        if (f.eyes === 'Open') fe += 0.15;
        else if (f.eyes === 'Droopy') fe -= 0.15;
        else if (f.eyes === 'Closed') fe -= 0.3;
        if (f.nod === 'Active') fe += 0.2;
        else if (f.nod === 'Light') fe += 0.05;
        else fe -= 0.05;
        if (f.brow === 'Raised') fe += 0.1;
        if (f.mouth === 'Open') fe += 0.1;
        energyScore += fe; weights += 1;
      }
      faceValue = weights > 0 ? energyScore / weights : 0.5;
      faceConfidence = Math.min(1, recent.length / 6);

      if (this.calibrator.calibrated) {
        const engDelta = this.calibrator.getDelta('engagement', recent[recent.length - 1].engagement || 50);
        const baseAdj = 0.5 + engDelta * 0.1;
        faceValue = faceValue * 0.7 + Math.max(0, Math.min(1, baseAdj)) * 0.3;
      }
      this.lastFaceEnergy = { value: faceValue, confidence: faceConfidence, timestamp: now };
    } else if (this.lastFaceEnergy) {
      const staleness = now - this.lastFaceEnergy.timestamp;
      const decayFactor = Math.max(0, 1 - staleness / this.FACE_DECAY_MS);
      faceValue = this.lastFaceEnergy.value;
      faceConfidence = this.lastFaceEnergy.confidence * decayFactor;
    }

    let sensorValue = null, sensorConfidence = 0;
    const movements = recentSensors.filter(s => s.accel)
      .map(s => Math.abs(Math.sqrt(s.accel.x ** 2 + s.accel.y ** 2 + s.accel.z ** 2) - 9.8));
    if (movements.length > 0) {
      const avgMov = movements.reduce((a, b) => a + b, 0) / movements.length;
      let movDelta = avgMov;
      if (this.calibrator.calibrated) movDelta = this.calibrator.getDelta('movement', avgMov);
      sensorValue = 0.5 + Math.min(0.3, Math.max(-0.3, movDelta * 0.1));
      sensorConfidence = 0.4;
    }

    const hrReadings = recentSensors.filter(s => s.hr && s.hr > 0).map(s => s.hr);
    if (hrReadings.length > 0) {
      const avgHR = hrReadings.reduce((a, b) => a + b, 0) / hrReadings.length;
      if (this.calibrator.calibrated && this.calibrator.baseline.heartRate) {
        const hrDelta = this.calibrator.getDelta('heartRate', avgHR);
        const hrContrib = 0.5 + Math.min(0.25, Math.max(-0.25, hrDelta * 0.1));
        if (sensorValue !== null) { sensorValue = (sensorValue + hrContrib) / 2; sensorConfidence = 0.6; }
        else { sensorValue = hrContrib; sensorConfidence = 0.5; }
      }
    }

    const intent = context?.intent || [];
    const isWindDown = intent.includes('sleep_prep') || intent.includes('unwind');
    let timeDrift = 0;
    if (isWindDown && this.stateHistory.length > 12) {
      timeDrift = -0.01 * (this.stateHistory.length * 5 / 60);
    }

    let value;
    if (faceValue !== null && sensorValue !== null) {
      const tot = faceConfidence + sensorConfidence;
      value = (faceValue * faceConfidence + sensorValue * sensorConfidence) / tot;
    } else if (faceValue !== null) value = faceValue;
    else if (sensorValue !== null) value = sensorValue;
    else value = 0.5;

    value = Math.max(0, Math.min(1, value + timeDrift));
    const confidence = Math.max(faceConfidence || 0, sensorConfidence || 0);
    let level;
    if (value < 0.25) level = 'low';
    else if (value < 0.55) level = 'medium';
    else if (value < 0.8) level = 'high';
    else level = 'overextended';
    return { level, value: Math.round(value * 100) / 100, confidence: Math.round(confidence * 100) / 100 };
  }

  _estimateImmersion(context) {
    const now = Date.now();
    const recent = this.faceHistory.slice(-6);
    const recentSensors = this.sensorHistory.slice(-10);
    const latestFaceTime = recent.length > 0 ? recent[recent.length - 1].t : 0;
    const faceFresh = recent.length > 0 && (now - latestFaceTime) < this.FACE_STALE_MS;
    const intent = context?.intent || [];
    const isWindDown = intent.includes('sleep_prep') || intent.includes('unwind');

    let faceValue = null, faceConfidence = 0;

    if (faceFresh) {
      let immScore = 0, weights = 0;
      const eyeStates = recent.map(f => f.eyes);
      const eyeCons = this._consistency(eyeStates);

      if (isWindDown) {
        const closedCount = eyeStates.filter(e => e === 'Closed' || e === 'Droopy').length;
        if (closedCount > eyeStates.length * 0.7 && eyeCons > 0.6) immScore += 0.8;
        else if (eyeStates.every(e => e === 'Open')) immScore += 0.5;
        else immScore += 0.4;
      } else {
        const openCount = eyeStates.filter(e => e === 'Open').length;
        immScore += (openCount / eyeStates.length) * 0.7;
      }
      weights += 1;

      const headPoses = recent.map(f => f.headPose);
      immScore += this._consistency(headPoses) * 0.6; weights += 1;

      const engValues = recent.map(f => f.engagement || 50);
      const engStd = this._std(engValues);
      immScore += Math.max(0, 1 - engStd / 20) * 0.7; weights += 1;

      const nods = recent.map(f => f.nod);
      const lightNods = nods.filter(n => n === 'Light').length;
      if (lightNods > nods.length * 0.4) { immScore += 0.6; weights += 1; }

      faceValue = weights > 0 ? immScore / weights : 0.4;
      faceConfidence = Math.min(1, recent.length / 6);
      this.lastFaceImmersion = { value: faceValue, confidence: faceConfidence, timestamp: now };
    } else if (this.lastFaceImmersion) {
      const staleness = now - this.lastFaceImmersion.timestamp;
      const decayFactor = Math.max(0, 1 - staleness / this.FACE_DECAY_MS);
      faceValue = this.lastFaceImmersion.value;
      faceConfidence = this.lastFaceImmersion.confidence * decayFactor;
    }

    let sensorValue = null, sensorConfidence = 0;
    const gyroMags = recentSensors.filter(s => s.gyro)
      .map(s => Math.sqrt(s.gyro.x ** 2 + s.gyro.y ** 2 + s.gyro.z ** 2));
    if (gyroMags.length > 3) {
      const avgGyro = gyroMags.reduce((a, b) => a + b, 0) / gyroMags.length;
      sensorValue = Math.max(0, 1 - avgGyro / 50) * 0.7;
      sensorConfidence = 0.35;
    }

    let value;
    if (faceValue !== null && sensorValue !== null) {
      const tot = faceConfidence + sensorConfidence;
      value = (faceValue * faceConfidence + sensorValue * sensorConfidence) / tot;
    } else if (faceValue !== null) value = faceValue;
    else if (sensorValue !== null) value = sensorValue;
    else value = 0.4;

    value = Math.max(0, Math.min(1, value));
    const confidence = Math.max(faceConfidence || 0, sensorConfidence || 0);
    let level;
    if (value < 0.2) level = 'detached';
    else if (value < 0.45) level = 'observing';
    else if (value < 0.7) level = 'engaged';
    else level = 'absorbed';
    return { level, value: Math.round(value * 100) / 100, confidence: Math.round(confidence * 100) / 100 };
  }

  _estimateTrajectory() {
    const history = this.stateHistory.slice(-18);
    if (history.length < 4) return { direction: 'flat', value: 0, confidence: 0 };
    const scores = history.map((h, i) => ({ x: i, y: h.energy * 0.4 + h.immersion * 0.6 }));
    const n = scores.length;
    const sumX = scores.reduce((s, p) => s + p.x, 0);
    const sumY = scores.reduce((s, p) => s + p.y, 0);
    const sumXY = scores.reduce((s, p) => s + p.x * p.y, 0);
    const sumX2 = scores.reduce((s, p) => s + p.x * p.x, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const meanY = sumY / n;
    const ssRes = scores.reduce((s, p) => s + (p.y - (meanY + slope * (p.x - sumX / n))) ** 2, 0);
    const ssTot = scores.reduce((s, p) => s + (p.y - meanY) ** 2, 0);
    const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
    const yVals = scores.map(s => s.y);
    const volatility = this._std(yVals);

    let direction;
    if (volatility > 0.15 && Math.abs(slope) < 0.02) direction = 'volatile';
    else if (slope > 0.01) direction = 'improving';
    else if (slope < -0.01) direction = 'declining';
    else direction = 'flat';

    return {
      direction, value: Math.round(slope * 1000) / 1000,
      volatility: Math.round(volatility * 100) / 100,
      confidence: Math.min(1, Math.abs(r2)), windowSize: history.length
    };
  }

  _getRawSignals() {
    const lastFace = this.faceHistory.length > 0 ? this.faceHistory[this.faceHistory.length - 1] : {};
    const recentFace = this.faceHistory.slice(-6);
    const avgEng = recentFace.length > 0
      ? recentFace.reduce((s, f) => s + (f.engagement || 50), 0) / recentFace.length : 50;
    return {
      currentEngagement: lastFace.engagement || 50,
      avgEngagement30s: Math.round(avgEng),
      eyes: lastFace.eyes || 'Unknown', headPose: lastFace.headPose || 'Unknown',
      nod: lastFace.nod || 'Unknown', brow: lastFace.brow || 'Unknown',
      faceReadings: this.faceHistory.length, sensorReadings: this.sensorHistory.length
    };
  }

  _consistency(arr) {
    if (arr.length === 0) return 0;
    const counts = {};
    arr.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    return Math.max(...Object.values(counts)) / arr.length;
  }

  _std(arr) {
    if (arr.length < 2) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
  }

  toJSON() { return { ...this.state }; }
}

// ============================================================
// FEEDBACK LOOP — Intervention effectiveness tracking
// ============================================================

class FeedbackLoop {
  constructor() {
    this.pending = [];
    this.records = [];
    this.stats = {};
    this.POST_DELAY = 60000;
  }

  registerIntervention(intervention, preState) {
    this.pending.push({
      intervention,
      preState: {
        energy: preState.energy.value, immersion: preState.immersion.value,
        trajectory: preState.trajectory.value, trajectoryDir: preState.trajectory.direction
      },
      registeredAt: Date.now(), measured: false
    });
  }

  tick(currentState) {
    const now = Date.now();
    const outcomes = [];
    for (const entry of this.pending) {
      if (entry.measured) continue;
      if (now - entry.registeredAt >= this.POST_DELAY) {
        const postState = {
          energy: currentState.energy.value, immersion: currentState.immersion.value,
          trajectory: currentState.trajectory.value, trajectoryDir: currentState.trajectory.direction
        };
        const outcome = this._score(entry.preState, postState, entry.intervention);
        entry.measured = true;
        const record = {
          intervention: {
            category: entry.intervention.category, intensity: entry.intervention.intensity,
            reason: entry.intervention.reason, text: entry.intervention.text, action: entry.intervention.action
          },
          preState: entry.preState, postState, outcome,
          timestamp: entry.registeredAt, measuredAt: now
        };
        this.records.push(record);
        this._updateStats(record);
        outcomes.push(record);
      }
    }
    this.pending = this.pending.filter(e => !e.measured);
    return outcomes;
  }

  _score(pre, post, intervention) {
    const cat = intervention.category;
    let score = 0, factors = [];

    if (cat === 'stimulate') {
      const eDelta = post.energy - pre.energy;
      const iDelta = post.immersion - pre.immersion;
      if (eDelta > 0.05) { score += 1; factors.push('energy_up'); }
      if (iDelta > 0.05) { score += 1; factors.push('immersion_up'); }
      if (post.trajectoryDir === 'improving' && pre.trajectoryDir !== 'improving') { score += 2; factors.push('trajectory_reversed'); }
      if (eDelta < -0.05) { score -= 1; factors.push('energy_dropped'); }
      if (iDelta < -0.1) { score -= 1; factors.push('immersion_dropped'); }
    } else if (cat === 'regulate') {
      const eDelta = post.energy - pre.energy;
      if (eDelta < -0.05 && post.energy > 0.25) { score += 1; factors.push('calmed'); }
      if (post.immersion > pre.immersion) { score += 1; factors.push('immersion_up'); }
      if (post.energy < 0.2) { score -= 1; factors.push('overcorrected'); }
    } else if (cat === 'transition') {
      if ((post.trajectoryDir === 'improving' || post.trajectoryDir === 'flat') &&
          (pre.trajectoryDir === 'declining' || pre.trajectoryDir === 'volatile')) { score += 2; factors.push('stabilized'); }
      if (post.trajectoryDir === 'declining') { score -= 1; factors.push('still_declining'); }
    } else if (cat === 'hold') {
      const iDelta = post.immersion - pre.immersion;
      if (iDelta >= -0.05) { score += 1; factors.push('maintained'); }
      if (post.trajectoryDir === 'improving') { score += 1; factors.push('improving'); }
      if (iDelta < -0.15) { score -= 1; factors.push('lost_immersion'); }
    }

    let outcome;
    if (score >= 2) outcome = 'positive';
    else if (score <= -1) outcome = 'negative';
    else outcome = 'neutral';
    return { outcome, score, factors };
  }

  _updateStats(record) {
    const cat = record.intervention.category;
    if (!this.stats[cat]) this.stats[cat] = { total: 0, positive: 0, neutral: 0, negative: 0 };
    this.stats[cat].total++;
    this.stats[cat][record.outcome.outcome]++;
  }

  toJSON() {
    return {
      pending: this.pending.length, completed: this.records.length,
      stats: this.stats,
      recentOutcomes: this.records.slice(-5).map(r => ({
        category: r.intervention.category, reason: r.intervention.reason,
        outcome: r.outcome.outcome, factors: r.outcome.factors
      }))
    };
  }
}

// ============================================================
// INTERVENTION ENGINE — Decision intelligence + cooldown + graduated response
// ============================================================

class InterventionEngine {
  constructor(feedback) {
    this.feedback = feedback;
    this.baseCooldown = 60000;
    this.lastInterventionTime = 0;
    this.consecutiveFailures = 0;
    this.observationCount = 0;
    this.lastObservedState = null;
    this.recentActions = [];
    this.MAX_RECENT = 5;

    this.nudgeLibrary = {
      stimulate: {
        subtle: [
          { text: null, action: 'music_shift_up', haptic: 'light' },
          { text: null, action: 'volume_up', haptic: null }
        ],
        moderate: [
          { text: 'Feel that?', action: 'music_shift_up', haptic: 'double' },
          { text: 'Wake up.', action: null, haptic: 'medium' }
        ],
        direct: [
          { text: 'You still with me?', action: 'skip_track', haptic: 'double' },
          { text: 'New energy incoming.', action: 'skip_track', haptic: 'medium' }
        ]
      },
      regulate: {
        subtle: [
          { text: null, action: 'volume_down', haptic: null },
          { text: null, action: 'music_shift_down', haptic: 'light' }
        ],
        moderate: [
          { text: 'Breathe.', action: 'volume_down', haptic: 'light' },
          { text: 'Slow down...', action: 'music_shift_down', haptic: 'light' }
        ],
        direct: [
          { text: 'Easy. Let it settle.', action: 'skip_track', haptic: 'light' }
        ]
      },
      transition: {
        subtle: [{ text: null, action: 'music_arc_align', haptic: null }],
        moderate: [
          { text: 'Shifting...', action: 'music_arc_align', haptic: 'light' },
          { text: 'New chapter.', action: 'skip_track', haptic: 'light' }
        ],
        direct: [{ text: 'Time to move.', action: 'skip_track', haptic: 'double' }]
      },
      hold: {
        subtle: [{ text: null, action: 'extend_track', haptic: null }],
        moderate: [{ text: 'Let go...', action: 'extend_track', haptic: null }]
      }
    };
  }

  decide(state, context, sessionElapsed) {
    const now = Date.now();
    const { energy, immersion, trajectory } = state;

    if (immersion.level === 'absorbed') {
      this.observationCount = 0;
      if (context.trackElapsed > 240) return this._makeDecision('hold', 'subtle', 'absorbed_extend', state);
      return null;
    }

    const effectiveCooldown = this.baseCooldown * Math.pow(1.5, this.consecutiveFailures);
    if (now - this.lastInterventionTime < effectiveCooldown) return null;
    if (!context.calibrated) return null;

    const intent = context.intent || [];
    const isWindDown = intent.includes('sleep_prep') || intent.includes('unwind') || intent.includes('escape');
    const isActive = intent.includes('energy') || intent.includes('focus');

    if (trajectory.direction === 'declining' && trajectory.confidence > 0.3) {
      this.observationCount++;
      if (this.observationCount <= 2) return null;
      else if (this.observationCount <= 4) {
        if (energy.level === 'low' && !isWindDown) return this._makeDecision('stimulate', 'subtle', 'declining_low_energy', state);
        else if (energy.level === 'high' || energy.level === 'overextended') return this._makeDecision('regulate', 'subtle', 'declining_overextended', state);
        else return this._makeDecision('transition', 'subtle', 'declining_mid', state);
      } else if (this.observationCount <= 6) {
        if (energy.level === 'low' && !isWindDown) return this._makeDecision('stimulate', 'moderate', 'declining_persistent_low', state);
        else return this._makeDecision('transition', 'moderate', 'declining_persistent', state);
      } else {
        if (!isWindDown) return this._makeDecision('stimulate', 'direct', 'declining_critical', state);
        else return this._makeDecision('hold', 'moderate', 'winddown_settling', state);
      }
    }

    if (trajectory.direction !== 'declining') this.observationCount = Math.max(0, this.observationCount - 1);

    if (context.arcEnergy !== undefined) {
      const mismatch = energy.value * 10 - context.arcEnergy;
      if (Math.abs(mismatch) > 3 && context.trackElapsed > 60) {
        if (mismatch > 0) return this._makeDecision('regulate', 'subtle', 'energy_above_arc', state);
        else if (!isWindDown) return this._makeDecision('stimulate', 'subtle', 'energy_below_arc', state);
      }
    }

    if (trajectory.direction === 'volatile' && trajectory.confidence > 0.3)
      return this._makeDecision('transition', 'subtle', 'volatile_stabilize', state);

    if (energy.level === 'low' && isActive && immersion.level !== 'engaged')
      return this._makeDecision('stimulate', 'moderate', 'low_energy_active_intent', state);

    if (immersion.level === 'engaged' && trajectory.direction === 'improving' && context.trackElapsed > 200)
      return this._makeDecision('hold', 'subtle', 'high_engagement_extend', state);

    if (immersion.level === 'detached' && context.trackElapsed > 90) {
      this.observationCount++;
      if (this.observationCount > 3) return this._makeDecision('stimulate', 'moderate', 'detached_persistent', state);
    }

    return null;
  }

  _makeDecision(category, intensity, reason, state) {
    const lib = this.nudgeLibrary[category]?.[intensity];
    if (!lib || lib.length === 0) return null;
    const recentTexts = this.recentActions.map(a => a.text);
    let pick = lib.find(n => !recentTexts.includes(n.text)) || lib[0];
    const lastAction = this.recentActions.length > 0 ? this.recentActions[this.recentActions.length - 1] : null;
    if (lastAction && lastAction.text === pick.text && lastAction.action === pick.action)
      pick = lib[Math.floor(Math.random() * lib.length)];

    const decision = {
      category, intensity, reason, text: pick.text, action: pick.action, haptic: pick.haptic,
      state: { energy: state.energy.level, immersion: state.immersion.level, trajectory: state.trajectory.direction },
      timestamp: Date.now()
    };

    this.lastInterventionTime = Date.now();
    this.recentActions.push(decision);
    if (this.recentActions.length > this.MAX_RECENT) this.recentActions.shift();
    if (this.feedback) this.feedback.registerIntervention(decision, state);
    return decision;
  }

  onFeedback(outcome) {
    if (outcome === 'negative') { this.consecutiveFailures = Math.min(5, this.consecutiveFailures + 2); }
    else if (outcome === 'positive') { this.consecutiveFailures = Math.max(0, this.consecutiveFailures - 2); }
    else { this.consecutiveFailures = Math.min(5, this.consecutiveFailures + 0.5); }
  }

  toJSON() {
    return {
      cooldown: this.baseCooldown * Math.pow(1.5, this.consecutiveFailures),
      consecutiveFailures: this.consecutiveFailures, observationCount: this.observationCount,
      recentActions: this.recentActions.slice(-3).map(a => ({ category: a.category, intensity: a.intensity, reason: a.reason, text: a.text }))
    };
  }
}

// ============================================================
// MUSIC BRAIN — Intent-to-sound intelligence + arc design + junk filter
// ============================================================

class MusicBrain {
  constructor() {
    this.sonicProfiles = {
      feel: {
        vibes: {
          electronic: ['melodic techno emotional', 'deep house vocal', 'lane 8 anjunadeep'],
          ambient: ['post rock crescendo', 'sigur ros atmospheric', 'emotional ambient'],
          bollywood: ['arijit singh unplugged', 'bollywood emotional hits', 'hindi heartbreak acoustic'],
          classical: ['chopin nocturne emotional', 'max richter on the nature of daylight', 'ludovico einaudi experience'],
          mixed: ['songs that hit different', 'emotional playlist late night', 'goosebump music feeling']
        },
        arcShape: 'wave', energyRange: [3, 8], trackDuration: { min: 180, max: 300 }
      },
      escape: {
        vibes: {
          electronic: ['ambient techno spacious', 'moderat atmospheric', 'four tet beautiful'],
          ambient: ['stars of the lid ambient', 'brian eno music for airports', 'ambient drone journey'],
          bollywood: ['indian classical night raga', 'flute meditation indian', 'santoor peaceful'],
          classical: ['debussy reverie', 'satie gymnopedies', 'arvo part spiegel'],
          mixed: ['ethereal soundscapes journey', 'otherworldly music escape', 'floating ambient dreamy']
        },
        arcShape: 'descent', energyRange: [1, 5], trackDuration: { min: 240, max: 360 }
      },
      energy: {
        vibes: {
          electronic: ['peak time techno', 'charlotte de witte', 'amelie lens set'],
          ambient: ['post rock build explosions in the sky', 'mogwai crescendo', 'god is an astronaut'],
          bollywood: ['bollywood party hits', 'punjabi bass', 'nucleya bass'],
          classical: ['hans zimmer epic', 'two steps from hell', 'vivaldi four seasons intense'],
          mixed: ['high energy dance', 'festival anthems', 'peak time mix']
        },
        arcShape: 'mountain', energyRange: [5, 10], trackDuration: { min: 180, max: 270 }
      },
      focus: {
        vibes: {
          electronic: ['minimal techno focus', 'lo-fi house study', 'deep focus electronic'],
          ambient: ['focus ambient instrumental', 'brian eno ambient study', 'concentration drone'],
          bollywood: ['indian instrumental focus', 'sitar ambient', 'tabla rhythm meditation'],
          classical: ['bach cello suite', 'glenn gould goldberg', 'piano focus classical'],
          mixed: ['deep work music', 'flow state playlist', 'concentration instrumental']
        },
        arcShape: 'plateau', energyRange: [3, 5], trackDuration: { min: 240, max: 360 }
      },
      unwind: {
        vibes: {
          electronic: ['downtempo electronic chill', 'bonobo chill', 'tycho sunset'],
          ambient: ['ambient relaxation evening', 'peaceful ambient instrumental', 'peaceful ambient night'],
          bollywood: ['bollywood lofi chill', 'hindi acoustic covers soft', 'prateek kuhad gentle'],
          classical: ['debussy clair de lune', 'nils frahm says', 'olafur arnalds gentle'],
          mixed: ['evening unwind playlist', 'decompress after work music', 'gentle chill evening']
        },
        arcShape: 'descent', energyRange: [2, 6], trackDuration: { min: 210, max: 300 }
      },
      sleep_prep: {
        vibes: {
          electronic: ['ambient sleep drone', 'sleep electronic gentle', 'binaural beats sleep'],
          ambient: ['sleep ambient music', 'max richter sleep', 'deep sleep soundscape'],
          bollywood: ['hindi lullaby gentle', 'indian flute sleep', 'peaceful indian night'],
          classical: ['chopin nocturne sleep', 'debussy sleep', 'gentle piano sleep'],
          mixed: ['drift off playlist', 'deep sleep music', 'peaceful night dissolve']
        },
        arcShape: 'fadeout', energyRange: [0, 3], trackDuration: { min: 300, max: 420 }
      }
    };

    this.junkPatterns = [
      /subscribe/i, /like.*comment/i, /\bshorts?\b/i, /#manifest/i,
      /full album/i, /compilation/i, /hours?\s+mix/i, /\b10\s*hr/i,
      /reaction/i, /review/i, /tutorial/i, /how to/i, /interview/i,
      /creative process/i, /behind the scenes/i, /podcast/i, /vlog/i,
      /asmr/i, /\brad(io)?\b/i, /live stream/i,
      /cricket sounds?/i, /\bcrickets?\b.*\bnight\b/i, /nature sounds?/i, /rain sounds?/i,
      /ocean sounds?/i, /forest sounds?/i, /thunder sounds?/i, /water sounds?/i,
      /bird sounds?/i, /campfire sounds?/i, /fireplace sounds?/i,
      /white noise/i, /brown noise/i, /pink noise/i,
      /\bambient sounds?\b/i, /sound ?scape.*nature/i, /nature.*sound ?scape/i,
      /\bsleep sounds?\b/i, /\brelaxing sounds?\b/i,
      /guided meditation/i, /hypnosis/i, /affirmation/i
    ];

    this.currentProfile = null;
    this.currentArc = null;
    this.arcAdaptations = 0;
  }

  initialize(context) {
    const intent = (context.intent || [])[0] || 'unwind';
    const sleepHours = parseFloat(context.sleep_in_hours) || 3;
    this.currentProfile = this.sonicProfiles[intent] || this.sonicProfiles.unwind;
    this.currentArc = this._designArc(intent, sleepHours);
    return { profile: intent, arcPoints: this.currentArc.length, sessionDuration: this.currentArc[this.currentArc.length - 1]?.t || 30 };
  }

  buildInitialQueries(context) {
    const intent = (context.intent || [])[0] || 'unwind';
    const vibe = (context.vibe || [])[0] || 'mixed';
    const profile = this.sonicProfiles[intent] || this.sonicProfiles.unwind;
    const vibeQueries = profile.vibes[vibe] || profile.vibes.mixed;
    const queries = [];
    const phases = this._getArcPhases();
    for (const phase of phases) {
      const baseQuery = vibeQueries[Math.floor(Math.random() * vibeQueries.length)];
      queries.push({
        query: baseQuery, phase: phase.name, energy: phase.energy, count: phase.trackCount,
        duration: profile.trackDuration.min + Math.random() * (profile.trackDuration.max - profile.trackDuration.min)
      });
    }
    return queries;
  }

  getNextSearchQuery(state, sessionElapsed) {
    if (!this.currentProfile || !this.currentArc) return null;
    const targetEnergy = this.getTargetEnergy(sessionElapsed);
    const energyDesc = this._energyToDescriptor(targetEnergy);
    const stateMod = this._stateToModifier(state);
    const vibeKeys = Object.keys(this.currentProfile.vibes);
    const randomVibe = this.currentProfile.vibes[vibeKeys[Math.floor(Math.random() * vibeKeys.length)]];
    const baseQuery = randomVibe[Math.floor(Math.random() * randomVibe.length)];
    return {
      query: `${baseQuery} ${energyDesc} ${stateMod}`.trim(),
      energy: Math.round(targetEnergy), duration: this.currentProfile.trackDuration.min,
      reason: `arc:${Math.round(targetEnergy)} state:${state.energy?.level || '?'}`
    };
  }

  interpretAction(action, state, sessionElapsed) {
    const targetEnergy = this.getTargetEnergy(sessionElapsed);
    switch (action) {
      case 'skip_track': return { command: 'skip' };
      case 'extend_track': return { command: 'extend', duration: 120 };
      case 'volume_up': return { command: 'volume', delta: +5 };
      case 'volume_down': return { command: 'volume', delta: -5 };
      case 'music_shift_up': { const q = this.getNextSearchQuery(state, sessionElapsed); if (q) q.energy = Math.min(10, q.energy + 2); return { command: 'search_and_play', query: q }; }
      case 'music_shift_down': { const q = this.getNextSearchQuery(state, sessionElapsed); if (q) q.energy = Math.max(1, q.energy - 2); return { command: 'search_and_play', query: q }; }
      case 'music_arc_align': { const q = this.getNextSearchQuery(state, sessionElapsed); return { command: 'search_and_play', query: q }; }
      default: return null;
    }
  }

  filterResults(results) {
    return results.filter(r => {
      const title = r.title || '';
      for (const pat of this.junkPatterns) { if (pat.test(title)) return false; }
      if (title.length < 10) return false;
      return true;
    });
  }

  getTargetEnergy(sessionMinutes) {
    if (!this.currentArc || this.currentArc.length < 2) return 5;
    const maxT = this.currentArc[this.currentArc.length - 1].t;
    const t = Math.min(sessionMinutes, maxT);
    for (let i = 1; i < this.currentArc.length; i++) {
      if (this.currentArc[i].t >= t) {
        const prev = this.currentArc[i - 1], curr = this.currentArc[i];
        const pct = (t - prev.t) / (curr.t - prev.t);
        return prev.energy + pct * (curr.energy - prev.energy);
      }
    }
    return this.currentArc[this.currentArc.length - 1].energy;
  }

  adaptArc(state, sessionElapsed) {
    if (!this.currentArc) return;
    if (state.trajectory?.direction === 'improving' && state.energy?.level === 'high') {
      for (let i = 0; i < this.currentArc.length; i++) {
        if (this.currentArc[i].t > sessionElapsed) { this.currentArc[i].t *= 1.1; break; }
      }
      this.arcAdaptations++;
    }
    if (state.trajectory?.direction === 'declining') {
      const currentTarget = this.getTargetEnergy(sessionElapsed);
      if (currentTarget > 4) {
        for (let i = 0; i < this.currentArc.length; i++) {
          if (this.currentArc[i].t > sessionElapsed)
            this.currentArc[i].energy = Math.max(this.currentProfile.energyRange[0], this.currentArc[i].energy - 1);
        }
        this.arcAdaptations++;
      }
    }
  }

  _designArc(intent, sleepHours) {
    const profile = this.sonicProfiles[intent] || this.sonicProfiles.unwind;
    const dur = Math.min(sleepHours * 25, 45);
    const [minE, maxE] = profile.energyRange;
    const pts = [];
    const gen = (fn) => { for (let i = 0; i <= 20; i++) { const p = i / 20; pts.push({ t: p * dur, energy: fn(p, minE, maxE) }); } };

    switch (profile.arcShape) {
      case 'wave': gen((p, mn, mx) => { const env = p < 0.6 ? p / 0.6 : 1 - (p - 0.6) / 0.4; const w = Math.sin(p * Math.PI * 3) * 0.3; return Math.max(mn, Math.min(mx, mn + (mx - mn) * (env * 0.7 + w * env))); }); break;
      case 'descent': gen((p, mn, mx) => Math.max(mn, mx - (mx - mn) * Math.pow(p, 0.7))); break;
      case 'mountain': gen((p, mn, mx) => { let e; if (p < 0.35) e = mn + (mx - mn) * (p / 0.35); else if (p < 0.6) e = mx; else e = mx - (mx - mn) * ((p - 0.6) / 0.4); return Math.max(mn, Math.min(mx, e)); }); break;
      case 'plateau': gen((p, mn, mx) => { const mid = (mn + mx) / 2; return Math.max(mn, Math.min(mx, mid + Math.sin(p * Math.PI * 2) * 0.5)); }); break;
      case 'fadeout': gen((p, mn, mx) => Math.max(0.5, mx * Math.pow(1 - p, 1.5))); break;
      default: gen((p, mn, mx) => { let e; if (p < 0.2) e = mn + (mx - mn) * p * 2.5; else if (p < 0.5) e = mx; else e = mx - (mx - mn) * ((p - 0.5) / 0.5); return Math.max(mn, Math.min(mx, e)); });
    }
    return pts;
  }

  _getArcPhases() {
    if (!this.currentArc || this.currentArc.length < 4) return [{ name: 'default', energy: 5, trackCount: 4 }];
    const third = Math.floor(this.currentArc.length / 3);
    const avg = (arr) => arr.reduce((s, p) => s + p.energy, 0) / arr.length;
    const phases = [
      { name: 'opening', energy: Math.round(avg(this.currentArc.slice(0, third))), trackCount: 3 },
      { name: 'core', energy: Math.round(avg(this.currentArc.slice(third, third * 2))), trackCount: 3 },
      { name: 'closing', energy: Math.round(avg(this.currentArc.slice(third * 2))), trackCount: 3 }
    ];
    if (phases[2].energy > 3) phases.push({ name: 'settle', energy: 2, trackCount: 2 });
    return phases;
  }

  _energyToDescriptor(energy) {
    if (energy >= 8) return 'powerful intense peak';
    if (energy >= 6) return 'upbeat driving';
    if (energy >= 4) return 'chill mellow';
    if (energy >= 2) return 'gentle calm relaxing';
    return 'ambient minimal sleep';
  }

  _stateToModifier(state) {
    if (!state || !state.energy) return '';
    if (state.energy.level === 'low' && state.immersion?.level === 'detached') return 'catchy popular';
    if (state.immersion?.level === 'absorbed') return 'deep atmospheric';
    if (state.trajectory?.direction === 'declining') return 'fresh new';
    return '';
  }

  toJSON() {
    return {
      profile: this.currentProfile ? Object.keys(this.sonicProfiles).find(k => this.sonicProfiles[k] === this.currentProfile) : null,
      arcAdaptations: this.arcAdaptations,
      arc: this.currentArc ? this.currentArc.map(p => ({ t: Math.round(p.t * 10) / 10, e: Math.round(p.energy * 10) / 10 })) : null
    };
  }
}

// ============================================================
// FACE ANALYZER — Extracts behavioral signals from face-api.js detections
// ============================================================

class FaceAnalyzer {
  constructor() {
    this.prevLandmarks = null;
    this.prevTimestamp = 0;
    this.headPoseHistory = [];
    this.MAX_POSE_HISTORY = 10;
  }

  // Takes face-api.js detection result, returns engine-compatible face data
  analyze(detection) {
    if (!detection || !detection.landmarks) {
      return { eyes: 'Unknown', headPose: 'Unknown', nod: 'None', brow: 'Normal', mouth: 'Closed', engagement: 30 };
    }

    const landmarks = detection.landmarks;
    const positions = landmarks.positions || landmarks._positions || [];
    if (positions.length < 68) {
      return { eyes: 'Unknown', headPose: 'Unknown', nod: 'None', brow: 'Normal', mouth: 'Closed', engagement: 30 };
    }

    const eyes = this._analyzeEyes(positions);
    const headPose = this._analyzeHeadPose(positions);
    const nod = this._analyzeNod(headPose);
    const brow = this._analyzeBrow(positions);
    const mouth = this._analyzeMouth(positions);
    const engagement = this._computeEngagement(detection, eyes, headPose, mouth);

    this.prevLandmarks = positions;
    this.prevTimestamp = Date.now();

    return { eyes, headPose: headPose.pose, nod, brow, mouth, engagement };
  }

  _analyzeEyes(pos) {
    // Eye Aspect Ratio (EAR) using 68-point landmarks
    // Left eye: 36-41, Right eye: 42-47
    const leftEAR = this._ear(pos[36], pos[37], pos[38], pos[39], pos[40], pos[41]);
    const rightEAR = this._ear(pos[42], pos[43], pos[44], pos[45], pos[46], pos[47]);
    const avgEAR = (leftEAR + rightEAR) / 2;

    if (avgEAR > 0.25) return 'Open';
    if (avgEAR > 0.17) return 'Droopy';
    return 'Closed';
  }

  _ear(p1, p2, p3, p4, p5, p6) {
    const dist = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    const vertical = dist(p2, p6) + dist(p3, p5);
    const horizontal = dist(p1, p4);
    return horizontal > 0 ? vertical / (2 * horizontal) : 0;
  }

  _analyzeHeadPose(pos) {
    // Simplified head pose from nose tip (30) relative to face bounding box
    const nose = pos[30];
    const leftCheek = pos[0];
    const rightCheek = pos[16];
    const chin = pos[8];
    const forehead = pos[27]; // bridge of nose top

    const faceWidth = Math.abs(rightCheek.x - leftCheek.x);
    const faceHeight = Math.abs(chin.y - forehead.y);

    // Pitch: nose position relative to face center vertically
    const faceCenterY = (forehead.y + chin.y) / 2;
    const pitchRatio = (nose.y - faceCenterY) / (faceHeight || 1);

    let pose;
    if (pitchRatio > 0.15) pose = 'Down';
    else if (pitchRatio < -0.05) pose = 'Up';
    else pose = 'Forward';

    return { pose, pitch: pitchRatio, faceHeight };
  }

  _analyzeNod(headPose) {
    this.headPoseHistory.push({ pitch: headPose.pitch, t: Date.now() });
    if (this.headPoseHistory.length > this.MAX_POSE_HISTORY) this.headPoseHistory.shift();

    if (this.headPoseHistory.length < 3) return 'None';

    // Compute pitch variance over recent history
    const pitches = this.headPoseHistory.map(h => h.pitch);
    const mean = pitches.reduce((a, b) => a + b, 0) / pitches.length;
    const variance = pitches.reduce((s, v) => s + (v - mean) ** 2, 0) / pitches.length;

    if (variance > 0.01) return 'Active';
    if (variance > 0.003) return 'Light';
    return 'None';
  }

  _analyzeBrow(pos) {
    // Distance from brow midpoint to eye center
    const leftBrowMid = { x: (pos[19].x + pos[20].x) / 2, y: (pos[19].y + pos[20].y) / 2 };
    const leftEyeCenter = { x: (pos[36].x + pos[39].x) / 2, y: (pos[36].y + pos[39].y) / 2 };
    const browDist = Math.abs(leftBrowMid.y - leftEyeCenter.y);
    const faceHeight = Math.abs(pos[8].y - pos[27].y);
    const browRatio = browDist / (faceHeight || 1);

    return browRatio > 0.12 ? 'Raised' : 'Normal';
  }

  _analyzeMouth(pos) {
    // Lip separation: top lip (62) to bottom lip (66)
    const topLip = pos[62];
    const bottomLip = pos[66];
    const mouthOpen = Math.abs(bottomLip.y - topLip.y);
    const faceHeight = Math.abs(pos[8].y - pos[27].y);
    const mouthRatio = mouthOpen / (faceHeight || 1);

    return mouthRatio > 0.06 ? 'Open' : 'Closed';
  }

  _computeEngagement(detection, eyes, headPose, mouth) {
    let score = 50; // baseline

    // Face detection confidence
    const confidence = detection.detection?._score || detection.score || 0.5;
    score += (confidence - 0.5) * 40; // higher confidence = more engaged

    // Eyes contribution
    if (eyes === 'Open') score += 10;
    else if (eyes === 'Droopy') score -= 5;
    else score -= 15;

    // Head pose — forward = engaged
    if (headPose.pose === 'Forward') score += 5;
    else if (headPose.pose === 'Down') score -= 10;

    // Expression if available
    if (detection.expressions) {
      const expr = detection.expressions;
      const maxExpr = Math.max(expr.happy || 0, expr.surprised || 0);
      score += maxExpr * 20;
      score -= (expr.angry || 0) * 10;
      score -= (expr.disgusted || 0) * 10;
    }

    return Math.max(0, Math.min(100, Math.round(score)));
  }
}

// ============================================================
// ME ENGINE — Main orchestrator (client-side)
// ============================================================

class MEEngine {
  constructor() {
    this.calibrator = new Calibrator();
    this.feedback = new FeedbackLoop();
    this.stateEstimator = new StateEstimator(this.calibrator);
    this.intervention = new InterventionEngine(this.feedback);
    this.musicBrain = new MusicBrain();

    this.active = false;
    this.sessionId = null;
    this.context = null;
    this.sessionStartTime = null;
    this.loopId = null;
    this.currentTrack = null;
    this.trackStartedAt = null;

    // Callback-based event system (replaces Node EventEmitter)
    this.handlers = { command: [], state: [], log: [], intervention: [] };
  }

  on(event, handler) { if (this.handlers[event]) this.handlers[event].push(handler); }
  _emit(event, data) { (this.handlers[event] || []).forEach(h => h(data)); }

  startSession(context) {
    this.context = context;
    this.active = true;
    this.sessionStartTime = Date.now();
    this.sessionId = 'S-' + Date.now().toString(36);
    this.calibrator.start();
    this.musicBrain.initialize(context);
    this.loopId = setInterval(() => this._tick(), 5000);

    this._emit('log', {
      type: 'engine_start',
      message: `Engine started — calibrating for ${this.calibrator.calibrationDuration / 1000}s`,
      context: { intent: context.intent, vibe: context.vibe, sleepHours: context.sleep_in_hours }
    });

    return { queries: this.musicBrain.buildInitialQueries(context), arc: this.musicBrain.currentArc };
  }

  stopSession() {
    this.active = false;
    if (this.loopId) { clearInterval(this.loopId); this.loopId = null; }
    const summary = this.getSessionSummary();
    this._emit('log', { type: 'engine_stop', message: 'Engine stopped', summary });
    return summary;
  }

  ingestFaceData(data) {
    if (!this.active) return;
    this.calibrator.addFaceData(data);
    this.stateEstimator.addFaceData(data);
  }

  ingestSensorData(data) {
    if (!this.active) return;
    this.calibrator.addSensorData(data);
    this.stateEstimator.addSensorData(data);
  }

  setCurrentTrack(track) {
    this.currentTrack = track;
    this.trackStartedAt = Date.now();
  }

  _tick() {
    if (!this.active) return;
    const now = Date.now();
    const sessionElapsed = (now - this.sessionStartTime) / 60000;

    const wasCalibrated = this.calibrator.calibrated;
    this.calibrator.tick();
    if (!wasCalibrated && this.calibrator.calibrated) {
      this._emit('log', { type: 'calibration_complete', message: 'Baseline calibration complete', baseline: this.calibrator.baseline });
      this._emit('state', { type: 'calibration', status: 'complete', baseline: this.calibrator.baseline });
    }

    const state = this.stateEstimator.estimate(this.context);

    this._emit('state', {
      type: 'state_update',
      state: { energy: state.energy, immersion: state.immersion, trajectory: state.trajectory },
      raw: state.raw,
      calibration: { calibrated: this.calibrator.calibrated, progress: this.calibrator.getProgress() },
      arc: { target: Math.round(this.musicBrain.getTargetEnergy(sessionElapsed) * 10) / 10, sessionMinutes: Math.round(sessionElapsed * 10) / 10 },
      cooldown: {
        remaining: Math.max(0, Math.round((this.intervention.lastInterventionTime + this.intervention.baseCooldown * Math.pow(1.5, this.intervention.consecutiveFailures) - now) / 1000)),
        failures: this.intervention.consecutiveFailures
      },
      timestamp: now
    });

    const feedbackOutcomes = this.feedback.tick(state);
    for (const outcome of feedbackOutcomes) {
      this.intervention.onFeedback(outcome.outcome.outcome);
      this._emit('log', { type: 'feedback', message: `"${outcome.intervention.reason}" scored: ${outcome.outcome.outcome}`, outcome });
    }

    const trackElapsed = this.trackStartedAt ? (now - this.trackStartedAt) / 1000 : 0;
    const decision = this.intervention.decide(state, {
      intent: this.context?.intent || [], calibrated: this.calibrator.calibrated,
      trackElapsed, arcEnergy: this.musicBrain.getTargetEnergy(sessionElapsed), sessionElapsed
    }, sessionElapsed);

    if (decision) this._executeDecision(decision, state, sessionElapsed);
    this.musicBrain.adaptArc(state, sessionElapsed);

    this._emit('log', {
      type: 'tick',
      state: { energy: state.energy.level, immersion: state.immersion.level, trajectory: state.trajectory.direction },
      engagement: state.raw.avgEngagement30s,
      arcTarget: Math.round(this.musicBrain.getTargetEnergy(sessionElapsed) * 10) / 10,
      interventionPending: this.feedback.pending.length,
      cooldown: this.intervention.toJSON().cooldown
    });
  }

  _executeDecision(decision, state, sessionElapsed) {
    this._emit('log', { type: 'intervention_fired', message: `[${decision.category}/${decision.intensity}] ${decision.reason}`, decision });
    this._emit('intervention', decision);

    if (decision.text) this._emit('command', { type: 'nudge', text: decision.text });
    if (decision.haptic) this._emit('command', { type: 'haptic', pattern: decision.haptic });
    if (decision.action) {
      const musicCommand = this.musicBrain.interpretAction(decision.action, state, sessionElapsed);
      if (musicCommand) this._emit('command', { type: 'music', ...musicCommand });
    }
  }

  getNextSearchQuery() {
    if (!this.active) return null;
    const state = this.stateEstimator.state;
    const sessionElapsed = (Date.now() - this.sessionStartTime) / 60000;
    return this.musicBrain.getNextSearchQuery(state, sessionElapsed);
  }

  filterYouTubeResults(results) { return this.musicBrain.filterResults(results); }

  getStatus() {
    return {
      active: this.active, sessionId: this.sessionId,
      calibration: this.calibrator.toJSON(), state: this.stateEstimator.toJSON(),
      intervention: this.intervention.toJSON(), feedback: this.feedback.toJSON(),
      music: this.musicBrain.toJSON(),
      uptime: this.sessionStartTime ? Math.round((Date.now() - this.sessionStartTime) / 1000) : 0
    };
  }

  getSessionSummary() {
    const feedback = this.feedback.toJSON();
    const state = this.stateEstimator.toJSON();
    return {
      sessionId: this.sessionId,
      duration: this.sessionStartTime ? Math.round((Date.now() - this.sessionStartTime) / 60000) : 0,
      calibration: this.calibrator.calibrated ? 'complete' : `${Math.round(this.calibrator.getProgress() * 100)}%`,
      finalState: state,
      interventions: { total: feedback.completed, stats: feedback.stats, recentOutcomes: feedback.recentOutcomes },
      arcAdaptations: this.musicBrain.arcAdaptations,
      profile: this.musicBrain.toJSON().profile
    };
  }
}

// ============================================================
// YOUTUBE SEARCH — Invidious (no key) + YouTube Data API (fallback)
// ============================================================

class YouTubeSearch {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.invidiousApis = [
      'https://vid.puffyan.us',
      'https://invidious.fdn.fr',
      'https://y.com.sb',
      'https://invidious.nerdvpn.de'
    ];

    this.fallbackTracks = {
      STIMULATE: [
        { id: 'wqBYHrw9_ys', title: 'Amelie Lens at Awakenings', artist: 'Amelie Lens' },
        { id: 'fBGSJ3sbivI', title: 'Charlotte de Witte Tomorrowland', artist: 'Charlotte de Witte' },
        { id: 'vqz8c4ZP3Wg', title: 'Boris Brejcha Grand Palais', artist: 'Boris Brejcha' },
        { id: 'ViwtNLUqkMY', title: 'Sara Landry Hard Techno', artist: 'Sara Landry' },
        { id: 'dEQsaO1JASg', title: 'Adam Beyer Drumcode', artist: 'Adam Beyer' }
      ],
      REGULATE: [
        { id: 'jfKfPfyJRdk', title: 'Lofi Hip Hop Radio', artist: 'Lofi Girl' },
        { id: 'DWcJFNfaw9c', title: 'Nils Frahm Says', artist: 'Nils Frahm' },
        { id: 'hlWiI4xGXNM', title: 'Nuvole Bianche', artist: 'Ludovico Einaudi' },
        { id: '77ZozI0rw7w', title: 'Interstellar', artist: 'Hans Zimmer' },
        { id: 'lE6RYpe9IT0', title: 'Relaxing Sleep Music', artist: 'Soothing Relaxation' }
      ],
      REFRESH: [
        { id: 'OPf0YbXqDm0', title: 'Uptown Funk', artist: 'Mark Ronson ft Bruno Mars' },
        { id: 'JGwWNGJdvx8', title: 'Shape of You', artist: 'Ed Sheeran' },
        { id: 'RgKAFK5djSk', title: 'See You Again', artist: 'Wiz Khalifa' },
        { id: 'kJQP7kiw5Fk', title: 'Despacito', artist: 'Luis Fonsi' },
        { id: 'CevxZvSJLk8', title: 'Roar', artist: 'Katy Perry' }
      ],
      TRANSITION: [
        { id: 'Dx5qFachd3A', title: 'Opus', artist: 'Eric Prydz' },
        { id: 'tKi9Z-f6qX4', title: 'Innerbloom', artist: 'RUFUS DU SOL' },
        { id: '0zGcUoRlhmw', title: 'Strobe', artist: 'deadmau5' },
        { id: 'WI4-HUn8dFc', title: 'Cola', artist: 'CamelPhat & Elderbrook' },
        { id: '_esYONwdKzA', title: 'Ben Bohmer Cercle', artist: 'Ben Bohmer' }
      ],
      EXPLORE: [
        { id: 'q4xKvHANqjk', title: 'Maria Tambien', artist: 'Khruangbin' },
        { id: 'kSE15tLBdOg', title: 'Ethno World', artist: 'Cafe De Anatolia' },
        { id: 'qA2_-2Fm7pk', title: 'Qawwali', artist: 'Nusrat Fateh Ali Khan' }
      ]
    };
  }

  async search(query, maxResults = 15) {
    // Try Invidious first (no API key needed)
    for (const api of this.invidiousApis) {
      try {
        const url = `${api}/api/v1/search?q=${encodeURIComponent(query)}&type=video&sort_by=relevance`;
        const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) continue;
        const data = await res.json();
        if (data && data.length > 0) {
          return data.filter(v => v.type === 'video').slice(0, maxResults).map(v => ({
            id: v.videoId, title: v.title, artist: v.author,
            thumb: v.videoThumbnails?.[4]?.url || `https://i.ytimg.com/vi/${v.videoId}/default.jpg`,
            duration: v.lengthSeconds
          }));
        }
      } catch (e) { continue; }
    }

    // Fallback: YouTube Data API
    if (this.apiKey) {
      try {
        const url = `https://www.googleapis.com/youtube/v3/search?part=snippet&q=${encodeURIComponent(query)}&type=video&maxResults=${maxResults}&key=${this.apiKey}`;
        const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
          const data = await res.json();
          return (data.items || []).map(item => ({
            id: item.id.videoId, title: item.snippet.title,
            artist: item.snippet.channelTitle,
            thumb: item.snippet.thumbnails?.default?.url || '',
            duration: 0
          }));
        }
      } catch (e) { /* fall through */ }
    }

    return [];
  }

  getFallbackTracks(intervention) {
    return this.fallbackTracks[intervention] || this.fallbackTracks.TRANSITION;
  }
}

// ============================================================
// SESSION LOGGER — localStorage + GitHub API commit
// ============================================================

class SessionLogger {
  constructor(options = {}) {
    this.sessionId = null;
    this.events = [];
    this.githubRepo = options.githubRepo || null;   // 'owner/repo'
    this.githubToken = options.githubToken || null;
    this.autoSaveInterval = null;
  }

  startSession(sessionId) {
    this.sessionId = sessionId;
    this.events = [];
    this.log('session_start', { sessionId, startedAt: new Date().toISOString(), userAgent: navigator.userAgent });
    // Auto-save to localStorage every 30s
    this.autoSaveInterval = setInterval(() => this._saveToLocalStorage(), 30000);
  }

  log(event, data) {
    const entry = { event, ...data, ts: new Date().toISOString(), sessionId: this.sessionId };
    this.events.push(entry);
  }

  async endSession(summary) {
    this.log('session_end', { summary });
    if (this.autoSaveInterval) { clearInterval(this.autoSaveInterval); this.autoSaveInterval = null; }
    this._saveToLocalStorage();
    if (this.githubRepo && this.githubToken) await this._saveToGitHub();
    return this.events;
  }

  _saveToLocalStorage() {
    try {
      const key = `me_session_${this.sessionId}`;
      localStorage.setItem(key, JSON.stringify(this.events));
      // Also maintain session index
      const index = JSON.parse(localStorage.getItem('me_sessions_index') || '[]');
      if (!index.includes(this.sessionId)) {
        index.push(this.sessionId);
        localStorage.setItem('me_sessions_index', JSON.stringify(index));
      }
    } catch (e) { /* storage full */ }
  }

  async _saveToGitHub() {
    if (!this.githubRepo || !this.githubToken) return;
    try {
      const filename = `data/sessions/${this.sessionId}.json`;
      const content = btoa(unescape(encodeURIComponent(JSON.stringify(this.events, null, 2))));
      const url = `https://api.github.com/repos/${this.githubRepo}/contents/${filename}`;
      await fetch(url, {
        method: 'PUT',
        headers: {
          'Authorization': `token ${this.githubToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: `Session ${this.sessionId} — ${this.events.length} events`,
          content: content
        })
      });
    } catch (e) { console.warn('GitHub save failed:', e); }
  }

  // Export all sessions from localStorage
  static getAllSessions() {
    const index = JSON.parse(localStorage.getItem('me_sessions_index') || '[]');
    return index.map(id => ({
      id,
      events: JSON.parse(localStorage.getItem(`me_session_${id}`) || '[]')
    }));
  }

  static exportAll() {
    const sessions = SessionLogger.getAllSessions();
    const blob = new Blob([JSON.stringify(sessions, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `me-sessions-${new Date().toISOString().slice(0, 10)}.json`;
    a.click(); URL.revokeObjectURL(url);
  }
}
