// M+E Intelligence Engine V4 — Browser Edition
// Built on V3 Session 59 engine (Kalman, circuit breaker, hysteresis, congruence gate)
// V4 changes: multi-intent blending, min play enforcement, direct intake wiring,
// structured analytics logger, artist variety tracking

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
// KALMAN FILTER — Lightweight 1D filter for browser (replaces EMA + hold-last-value)
// Based on kalmanjs (MIT) — principled uncertainty growth when measurements are missing
// ============================================================

class KalmanFilter {
  constructor(options = {}) {
    this.R = options.R || 1;    // Measurement noise (higher = trust measurements less)
    this.Q = options.Q || 0.01; // Process noise (higher = state changes faster)
    this.x = options.x0 || 0.5; // State estimate
    this.P = options.P0 || 1;   // Estimate covariance (uncertainty)
  }

  // Predict step: state propagates, uncertainty grows
  predict() {
    // State: x_k = x_{k-1} (near-constant model)
    // Covariance: P_k = P_{k-1} + Q
    this.P = this.P + this.Q;
    return this.x;
  }

  // Update step: incorporate a measurement
  update(measurement, measurementNoise) {
    const R = measurementNoise !== undefined ? measurementNoise : this.R;
    // Kalman gain: how much to trust the measurement vs the prediction
    const K = this.P / (this.P + R);
    // Update state: blend prediction with measurement
    this.x = this.x + K * (measurement - this.x);
    // Update covariance: uncertainty shrinks after measurement
    this.P = (1 - K) * this.P;
    return this.x;
  }

  // Get current confidence (inverse of uncertainty, 0-1 scale)
  getConfidence() {
    // Map covariance to 0-1: low P = high confidence
    return Math.max(0, Math.min(1, 1 / (1 + this.P)));
  }

  toJSON() { return { x: this.x, P: this.P, confidence: this.getConfidence() }; }
}

// ============================================================
// STATE ESTIMATOR — Kalman-filtered multi-signal fusion with hysteresis
// Phase 2a: Kalman replaces EMA + hold-last-value
// Phase 2b: Hysteresis dead bands on state boundaries
// Phase 2c: Separate energy/immersion feature extraction
// ============================================================

class StateEstimator {
  constructor(calibrator) {
    this.calibrator = calibrator;
    this.faceHistory = [];
    this.sensorHistory = [];
    this.stateHistory = [];

    // Kalman filters for each dimension
    // Q = process noise (how fast true state changes between ticks)
    // R = default measurement noise (overridden per-measurement by confidence)
    // Q = process noise. Higher = state can change faster between ticks.
    // V4 fix: Q was 0.005/0.003 → flatline in 31-min pilot. State MUST respond to measurements.
    // With MediaPipe (richer signal, more frequent), these values let the filter breathe.
    this.kf = {
      energy:    new KalmanFilter({ Q: 0.03, R: 0.3, x0: 0.5, P0: 1 }),    // Was 0.005 → flatlined
      immersion: new KalmanFilter({ Q: 0.02, R: 0.35, x0: 0.4, P0: 1 }),   // Was 0.003 → flatlined
      movement:  new KalmanFilter({ Q: 0.04, R: 0.2, x0: 0.5, P0: 1 })     // Was 0.02 → adequate but bump slightly
    };

    this.state = {
      energy: { level: 'medium', value: 0.5, confidence: 0 },
      immersion: { level: 'observing', value: 0.4, confidence: 0 },
      trajectory: { direction: 'flat', value: 0, confidence: 0 },
      raw: {}
    };

    // Phase 2b: Hysteresis — current levels + hold timers
    this._currentEnergyLevel = 'medium';
    this._currentImmersionLevel = 'observing';
    this._energyLevelHoldUntil = 0;   // timestamp — no transition before this
    this._immersionLevelHoldUntil = 0;
    this.MIN_LEVEL_HOLD_MS = 15000;   // 3 ticks minimum hold after any transition

    this.MAX_FACE_HISTORY = 12;
    this.MAX_SENSOR_HISTORY = 60;
    this.MAX_STATE_HISTORY = 60;

    // V4: Music interaction signals (W-6: signals beyond biometrics)
    this.musicInteractions = [];  // { type: 'skip'|'complete'|'moment', t, trackElapsed }
    this.MAX_MUSIC_INTERACTIONS = 20;

    // V4: Signal-scarcity detection (W-2: design for 5 contexts)
    this._lastFaceTime = 0;
    this._signalRegime = 'unknown'; // 'rich' | 'sparse' | 'absent'
    this._regimeDetectedAt = 0;
  }

  addFaceData(data) {
    this.faceHistory.push({ ...data, t: Date.now() });
    if (this.faceHistory.length > this.MAX_FACE_HISTORY) this.faceHistory.shift();
  }

  addSensorData(data) {
    this.sensorHistory.push({ ...data, t: Date.now() });
    if (this.sensorHistory.length > this.MAX_SENSOR_HISTORY) this.sensorHistory.shift();
  }

  // V4: Ingest music interaction events as state signals (W-6)
  addMusicInteraction(type, trackElapsed) {
    this.musicInteractions.push({ type, t: Date.now(), trackElapsed: trackElapsed || 0 });
    if (this.musicInteractions.length > this.MAX_MUSIC_INTERACTIONS) this.musicInteractions.shift();
  }

  estimate(context) {
    const now = Date.now();

    // === PREDICT: all filters propagate forward, uncertainty grows ===
    this.kf.energy.predict();
    this.kf.immersion.predict();
    this.kf.movement.predict();

    // === DETECT SIGNAL REGIME (W-2) ===
    this._detectSignalRegime(now);

    // === UPDATE: incorporate available measurements ===
    this._updateFromFace(now, context);
    this._updateFromSensors(now, context);
    this._updateFromMusicInteraction(now);

    // === TIME-BASED UNCERTAINTY GROWTH (W-4) ===
    // If no face data for >60s, actively grow uncertainty — state can't be flat forever
    this._applyTimeDrift(now);

    // === READ STATE from filters ===
    const energyVal = Math.max(0, Math.min(1, this.kf.energy.x));
    const immersionVal = Math.max(0, Math.min(1, this.kf.immersion.x));
    const energyConf = this.kf.energy.getConfidence();
    const immersionConf = this.kf.immersion.getConfidence();

    // Phase 2b: Hysteresis level classification
    const energyLevel = this._classifyEnergyWithHysteresis(energyVal, now);
    const immersionLevel = this._classifyImmersionWithHysteresis(immersionVal, now);

    const energy = { level: energyLevel, value: Math.round(energyVal * 100) / 100, confidence: Math.round(energyConf * 100) / 100 };
    const immersion = { level: immersionLevel, value: Math.round(immersionVal * 100) / 100, confidence: Math.round(immersionConf * 100) / 100 };

    this.stateHistory.push({ t: now, energy: energy.value, immersion: immersion.value });
    if (this.stateHistory.length > this.MAX_STATE_HISTORY) this.stateHistory.shift();

    const trajectory = this._estimateTrajectory();
    this.state = { energy, immersion, trajectory, raw: this._getRawSignals(), timestamp: now };
    return this.state;
  }

  // Phase 2c: Extract ENERGY features from face data (separate from immersion)
  _updateFromFace(now, context) {
    const recent = this.faceHistory.slice(-6);
    if (recent.length === 0) return; // No face data → predict-only, uncertainty grows naturally

    const latestFaceTime = recent[recent.length - 1].t;
    const faceAge = now - latestFaceTime;
    if (faceAge > 30000) return; // Stale face → predict-only (Kalman handles uncertainty growth)

    const latest = recent[recent.length - 1];

    // --- ENERGY observation: activity indicators ---
    // Use engagement score (now expression-free) as base, plus structural modifiers
    const engValues = recent.map(f => (f.engagement || 50) / 100);
    const avgEng = engValues.reduce((a, b) => a + b, 0) / engValues.length;
    let energyObs = avgEng;

    // Categorical modifiers (adjust, don't drive)
    if (latest.nod === 'Active') energyObs += 0.10;
    else if (latest.nod === 'Light') energyObs += 0.04;
    if (latest.mouth === 'Open') energyObs += 0.05;
    if (latest.eyes === 'Closed') energyObs -= 0.15;
    else if (latest.eyes === 'Droopy') energyObs -= 0.06;

    // Calibrator delta (baseline-relative, D-047)
    if (this.calibrator.calibrated) {
      const engDelta = this.calibrator.getDelta('engagement', latest.engagement || 50);
      energyObs += engDelta * 0.10;
    }

    energyObs = Math.max(0, Math.min(1, energyObs));

    // Measurement noise = f(1/confidence). Detection confidence → filter trust
    const detConf = latest.detectionConfidence || 0.5;
    const faceR = 0.1 + (1 - detConf) * 0.8; // High confidence → R=0.1, low → R=0.9
    this.kf.energy.update(energyObs, faceR);

    // --- IMMERSION observation: stability indicators (independent from energy) ---
    const engStd = this._std(engValues);
    // High avg + low variance = deeply immersed
    let immObs = avgEng * 0.5 + Math.max(0, 0.4 - engStd) * 0.6;

    // Eye consistency
    const eyeStates = recent.map(f => f.eyes);
    const openCount = eyeStates.filter(e => e === 'Open').length;
    immObs += (openCount / eyeStates.length) * 0.08;

    // Head consistency (staying put = immersed)
    const headPoses = recent.map(f => f.headPose);
    const headCons = this._consistency(headPoses);
    immObs += headCons * 0.08;

    // Rhythmic nodding (immersion signal in music context)
    const nods = recent.map(f => f.nod);
    const lightNods = nods.filter(n => n === 'Light' || n === 'Active').length;
    if (lightNods > nods.length * 0.3) immObs += 0.08;

    // Wind-down context: closed/droopy eyes + consistency = immersed in rest
    const intent = context?.intent || [];
    const isWindDown = intent.includes('sleep_prep') || intent.includes('unwind');
    if (isWindDown) {
      const closedCount = eyeStates.filter(e => e === 'Closed' || e === 'Droopy').length;
      if (closedCount > eyeStates.length * 0.5 && this._consistency(eyeStates) > 0.5) immObs += 0.12;
    }

    immObs = Math.max(0, Math.min(1, immObs));
    this.kf.immersion.update(immObs, faceR * 1.1); // Slightly higher noise for immersion
  }

  _updateFromSensors(now, context) {
    const recentSensors = this.sensorHistory.slice(-10);
    if (recentSensors.length === 0) return;

    // Movement from accelerometer
    const movements = recentSensors.filter(s => s.accel)
      .map(s => Math.abs(Math.sqrt(s.accel.x ** 2 + s.accel.y ** 2 + s.accel.z ** 2) - 9.8));
    if (movements.length > 0) {
      const avgMov = movements.reduce((a, b) => a + b, 0) / movements.length;
      let movNorm = avgMov;
      if (this.calibrator.calibrated) movNorm = this.calibrator.getDelta('movement', avgMov);
      const movValue = 0.5 + Math.min(0.35, Math.max(-0.35, movNorm * 0.15));

      // Burst detection
      const maxMov = Math.max(...movements);
      const burstBonus = (maxMov > avgMov * 2.5 && maxMov > 0.5) ? 0.1 : 0;

      this.kf.movement.update(Math.min(1, movValue + burstBonus), 0.2);
      // Movement contributes to energy estimate
      this.kf.energy.update(Math.min(1, movValue + burstBonus), 0.4); // Higher R — sensor is secondary to face for energy
    }

    // Gyro → body engagement
    const gyroMags = recentSensors.filter(s => s.gyro)
      .map(s => Math.sqrt(s.gyro.x ** 2 + s.gyro.y ** 2 + s.gyro.z ** 2));
    if (gyroMags.length > 0) {
      const avgGyro = gyroMags.reduce((a, b) => a + b, 0) / gyroMags.length;
      if (avgGyro > 5) {
        this.kf.energy.update(Math.min(1, this.kf.energy.x + 0.06), 0.5);
      }
    }

    // Heart rate → energy contribution
    const hrReadings = recentSensors.filter(s => s.hr && s.hr > 0).map(s => s.hr);
    if (hrReadings.length > 0 && this.calibrator.calibrated && this.calibrator.baseline?.heartRate) {
      const avgHR = hrReadings.reduce((a, b) => a + b, 0) / hrReadings.length;
      const hrDelta = this.calibrator.getDelta('heartRate', avgHR);
      const hrValue = 0.5 + Math.min(0.3, Math.max(-0.3, hrDelta * 0.12));
      this.kf.energy.update(hrValue, 0.5); // HR is noisy in nightlife, high R
    }

    // Low gyro → immersion contribution (stillness = absorption)
    if (gyroMags.length > 3) {
      const avgGyro = gyroMags.reduce((a, b) => a + b, 0) / gyroMags.length;
      const stillness = Math.max(0, 1 - avgGyro / 50) * 0.7;
      this.kf.immersion.update(stillness, 0.6); // Secondary signal, high noise
    }
  }

  // V4: Detect signal regime — rich / sparse / absent (W-2)
  _detectSignalRegime(now) {
    const lastFace = this.faceHistory.length > 0 ? this.faceHistory[this.faceHistory.length - 1] : null;
    const faceAge = lastFace ? (now - lastFace.t) : Infinity;

    // Check face read frequency over last 2 minutes
    const twoMinAgo = now - 120000;
    const recentFaceCount = this.faceHistory.filter(f => f.t > twoMinAgo).length;

    let regime;
    if (recentFaceCount >= 20) regime = 'rich';       // ~10+ reads per min
    else if (recentFaceCount >= 3) regime = 'sparse';  // Some reads but gaps
    else regime = 'absent';                             // <3 in 2 min

    if (regime !== this._signalRegime) {
      this._signalRegime = regime;
      this._regimeDetectedAt = now;
    }
  }

  // V4: Music interaction → state signal (W-6: signals beyond biometrics)
  _updateFromMusicInteraction(now) {
    const twoMinAgo = now - 120000;
    const recent = this.musicInteractions.filter(m => m.t > twoMinAgo);
    if (recent.length === 0) return;

    for (const interaction of recent) {
      if (interaction._processed) continue;
      interaction._processed = true;

      if (interaction.type === 'skip') {
        // Skip = dissatisfaction. Slight energy dip + immersion drop
        this.kf.immersion.update(Math.max(0, this.kf.immersion.x - 0.08), 0.4);
      } else if (interaction.type === 'complete') {
        // Natural track completion = satisfaction. Immersion boost
        this.kf.immersion.update(Math.min(1, this.kf.immersion.x + 0.05), 0.35);
      } else if (interaction.type === 'moment') {
        // "I feel something" press = peak experience signal
        this.kf.energy.update(Math.min(1, this.kf.energy.x + 0.10), 0.25);    // Strong signal
        this.kf.immersion.update(Math.min(1, this.kf.immersion.x + 0.12), 0.25);
      }
    }
  }

  // V4: Time-based uncertainty growth (W-4: "I don't know" ≠ "do nothing")
  _applyTimeDrift(now) {
    const lastFace = this.faceHistory.length > 0 ? this.faceHistory[this.faceHistory.length - 1] : null;
    const faceAge = lastFace ? (now - lastFace.t) / 1000 : Infinity;

    if (faceAge > 60) {
      // No face for >60s: grow uncertainty — covariance increases beyond normal Q
      // This prevents the Kalman from being confidently wrong about a stale state
      const extraUncertainty = Math.min(0.05, (faceAge - 60) / 1000 * 0.01);
      this.kf.energy.P += extraUncertainty;
      this.kf.immersion.P += extraUncertainty;
    }

    // In sparse/absent regime, gently drift energy toward arc target (W-4)
    // The system should follow the arc when it can't see the user
    if (this._signalRegime === 'absent' || this._signalRegime === 'sparse') {
      // Only drift if we have arc context
      if (this._arcTarget !== undefined) {
        const arcNorm = this._arcTarget / 10; // arc is 0-10, energy is 0-1
        const gap = arcNorm - this.kf.energy.x;
        // Gentle drift: 2% of gap per tick toward arc, high noise (uncertain)
        if (Math.abs(gap) > 0.05) {
          this.kf.energy.update(this.kf.energy.x + gap * 0.02, 0.8);
        }
      }
    }
  }

  // Phase 2b: Hysteresis on energy levels — dead bands prevent oscillation
  _classifyEnergyWithHysteresis(value, now) {
    // Dead band thresholds: enter at one value, exit at another
    const thresholds = {
      low:          { enterBelow: 0.27, exitAbove: 0.33 },  // Enter low < 0.27, exit low > 0.33
      medium:       { enterBelow: 0.52, exitAbove: 0.58 },  // Enter medium < 0.52, exit medium > 0.58
      high:         { enterBelow: 0.72, exitAbove: 0.78 }   // Enter high < 0.72, exit high > 0.78
    };

    // Hold timer: no transition allowed within MIN_LEVEL_HOLD_MS of last transition
    if (now < this._energyLevelHoldUntil) return this._currentEnergyLevel;

    let newLevel;
    const current = this._currentEnergyLevel;

    // Determine new level based on current level + hysteresis
    if (current === 'low') {
      newLevel = value > thresholds.low.exitAbove ? 'medium' : 'low';
    } else if (current === 'medium') {
      if (value < thresholds.low.enterBelow) newLevel = 'low';
      else if (value > thresholds.medium.exitAbove) newLevel = 'high';
      else newLevel = 'medium';
    } else if (current === 'high') {
      if (value < thresholds.medium.enterBelow) newLevel = 'medium';
      else if (value > thresholds.high.exitAbove) newLevel = 'overextended';
      else newLevel = 'high';
    } else { // overextended
      newLevel = value < thresholds.high.enterBelow ? 'high' : 'overextended';
    }

    if (newLevel !== current) {
      this._currentEnergyLevel = newLevel;
      this._energyLevelHoldUntil = now + this.MIN_LEVEL_HOLD_MS;
    }
    return this._currentEnergyLevel;
  }

  // Phase 2b: Hysteresis on immersion levels
  _classifyImmersionWithHysteresis(value, now) {
    const thresholds = {
      detached:  { enterBelow: 0.17, exitAbove: 0.23 },
      observing: { enterBelow: 0.42, exitAbove: 0.48 },
      engaged:   { enterBelow: 0.67, exitAbove: 0.73 }
    };

    if (now < this._immersionLevelHoldUntil) return this._currentImmersionLevel;

    let newLevel;
    const current = this._currentImmersionLevel;

    if (current === 'detached') {
      newLevel = value > thresholds.detached.exitAbove ? 'observing' : 'detached';
    } else if (current === 'observing') {
      if (value < thresholds.detached.enterBelow) newLevel = 'detached';
      else if (value > thresholds.observing.exitAbove) newLevel = 'engaged';
      else newLevel = 'observing';
    } else if (current === 'engaged') {
      if (value < thresholds.observing.enterBelow) newLevel = 'observing';
      else if (value > thresholds.engaged.exitAbove) newLevel = 'absorbed';
      else newLevel = 'engaged';
    } else { // absorbed
      newLevel = value < thresholds.engaged.enterBelow ? 'engaged' : 'absorbed';
    }

    if (newLevel !== current) {
      this._currentImmersionLevel = newLevel;
      this._immersionLevelHoldUntil = now + this.MIN_LEVEL_HOLD_MS;
    }
    return this._currentImmersionLevel;
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
      faceReadings: this.faceHistory.length, sensorReadings: this.sensorHistory.length,
      signalRegime: this._signalRegime,
      musicInteractions: this.musicInteractions.length,
      kalman: {
        energy: this.kf.energy.toJSON(),
        immersion: this.kf.immersion.toJSON(),
        movement: this.kf.movement.toJSON()
      }
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
      if (eDelta > 0.03) { score += 1; factors.push('energy_up'); }
      if (iDelta > 0.03) { score += 1; factors.push('immersion_up'); }
      if (post.trajectoryDir === 'improving' && pre.trajectoryDir !== 'improving') { score += 2; factors.push('trajectory_reversed'); }
      if (eDelta < -0.03) { score -= 1; factors.push('energy_dropped'); }
      if (iDelta < -0.08) { score -= 1; factors.push('immersion_dropped'); }
    } else if (cat === 'regulate') {
      const eDelta = post.energy - pre.energy;
      if (eDelta < -0.03 && post.energy > 0.2) { score += 1; factors.push('calmed'); }
      if (post.immersion >= pre.immersion - 0.02) { score += 1; factors.push('immersion_held'); }
      if (post.energy < 0.15) { score -= 1; factors.push('overcorrected'); }
    } else if (cat === 'transition') {
      if ((post.trajectoryDir === 'improving' || post.trajectoryDir === 'flat') &&
          (pre.trajectoryDir === 'declining' || pre.trajectoryDir === 'volatile')) { score += 2; factors.push('stabilized'); }
      if (post.trajectoryDir === 'declining') { score -= 1; factors.push('still_declining'); }
    } else if (cat === 'hold') {
      const iDelta = post.immersion - pre.immersion;
      if (iDelta >= -0.03) { score += 1; factors.push('maintained'); }
      if (post.trajectoryDir === 'improving') { score += 1; factors.push('improving'); }
      if (iDelta < -0.10) { score -= 1; factors.push('lost_immersion'); }
    }

    // Phase 2d: Trajectory-adjusted attribution
    // If pre-intervention trajectory was already moving in the "right" direction,
    // discount positive outcomes — the trajectory might have done the work, not the intervention
    if (score > 0) {
      const trajectoryAlreadyHelping =
        (cat === 'stimulate' && pre.trajectoryDir === 'improving') ||
        (cat === 'regulate' && pre.trajectoryDir === 'declining') ||
        (cat === 'hold' && (pre.trajectoryDir === 'flat' || pre.trajectoryDir === 'improving'));
      if (trajectoryAlreadyHelping) {
        score = Math.ceil(score * 0.5); // 50% discount — trajectory was already doing this
        factors.push('trajectory_discount');
      }
    }
    // If trajectory was AGAINST the intervention goal and we still got positive, boost
    if (score > 0) {
      const trajectoryFighting =
        (cat === 'stimulate' && pre.trajectoryDir === 'declining') ||
        (cat === 'regulate' && pre.trajectoryDir === 'improving');
      if (trajectoryFighting) {
        score = Math.min(4, score + 1); // Bonus — intervention overcame adverse trajectory
        factors.push('trajectory_boost');
      }
    }

    let outcome;
    if (score >= 1) outcome = 'positive';
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
    this.baseCooldown = 45000; // 45s base
    this.MAX_COOLDOWN = 180000; // hard cap: 3 min
    this.lastInterventionTime = 0;
    this.consecutiveFailures = 0;
    this.lastFailureDecayTime = Date.now();
    this.observationCount = 0;
    this.lastObservedState = null;
    this.recentActions = [];
    this.MAX_RECENT = 5;

    // Phase 1a: Circuit breaker — 3 strikes on same reason → blocked (with recovery)
    this.reasonFailures = {};    // { reason: { neutral: N, negative: N } }
    this.blockedReasons = new Set();
    this.blockedAt = {};         // { reason: timestamp } — for recovery timing
    this.REASON_STRIKE_LIMIT = 3; // consecutive neutral/negative on same reason → block
    this.RECOVERY_TIME = 600000;  // V4: 10 min silence → unblock and retry (was permanent)

    // Phase 1b: Hourly intervention cap
    this.interventionTimestamps = []; // timestamps of all interventions
    this.MAX_PER_HOUR = 5;           // hard ceiling regardless of tolerance

    // Phase 1b: Consecutive neutral/negative suppression
    this.consecutiveNeutralNegative = 0;
    this.suppressionUntil = 0; // timestamp — when >0 and now < this, only hold/safety

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

    // === GATE: Absorption — only extend track, no interventions ===
    if (immersion.level === 'absorbed') {
      this.observationCount = 0;
      if (context.trackElapsed > 240) return this._makeDecision('hold', 'subtle', 'absorbed_extend', state);
      return null;
    }

    // === GATE: Hourly cap — prune old timestamps, enforce ceiling ===
    this.interventionTimestamps = this.interventionTimestamps.filter(t => now - t < 3600000);
    if (this.interventionTimestamps.length >= this.MAX_PER_HOUR) return null;

    // === GATE: 15-min suppression after 3 consecutive neutral/negative ===
    if (now < this.suppressionUntil) {
      // Only hold/safety interventions allowed during suppression
      if (immersion.level === 'engaged' && context.trackElapsed > 200)
        return this._makeDecision('hold', 'subtle', 'suppression_hold', state);
      return null;
    }

    // Time-decay consecutive failures (every 180s of inactivity, decay by 0.15)
    const sinceLastDecay = now - this.lastFailureDecayTime;
    if (sinceLastDecay > 180000 && this.consecutiveFailures > 0) {
      this.consecutiveFailures = Math.max(0, this.consecutiveFailures - 0.15 * Math.floor(sinceLastDecay / 180000));
      this.lastFailureDecayTime = now;
    }
    const effectiveCooldown = Math.min(this.MAX_COOLDOWN, this.baseCooldown * Math.pow(1.3, this.consecutiveFailures));
    if (now - this.lastInterventionTime < effectiveCooldown) return null;
    if (!context.calibrated) return null;

    const intent = context.intent || [];
    const isWindDown = intent.includes('sleep_prep') || intent.includes('unwind') || intent.includes('escape');
    const isActive = intent.includes('energy') || intent.includes('focus');

    // === CANDIDATE GENERATION (same logic, but candidates go through circuit breaker) ===
    let candidate = null;

    if (trajectory.direction === 'declining' && trajectory.confidence > 0.3) {
      this.observationCount++;
      if (this.observationCount <= 3) { /* minimum 3 ticks (15s) observation before acting */ }
      else if (this.observationCount <= 5) {
        if (energy.level === 'low' && !isWindDown) candidate = { cat: 'stimulate', int: 'subtle', reason: 'declining_low_energy' };
        else if (energy.level === 'high' || energy.level === 'overextended') candidate = { cat: 'regulate', int: 'subtle', reason: 'declining_overextended' };
        else candidate = { cat: 'transition', int: 'subtle', reason: 'declining_mid' };
      } else if (this.observationCount <= 8) {
        if (energy.level === 'low' && !isWindDown) candidate = { cat: 'stimulate', int: 'moderate', reason: 'declining_persistent_low' };
        else candidate = { cat: 'transition', int: 'moderate', reason: 'declining_persistent' };
      } else {
        if (!isWindDown) candidate = { cat: 'stimulate', int: 'direct', reason: 'declining_critical' };
        else candidate = { cat: 'hold', int: 'moderate', reason: 'winddown_settling' };
      }
    }

    if (!candidate && trajectory.direction !== 'declining') this.observationCount = Math.max(0, this.observationCount - 1);

    if (!candidate && context.arcEnergy !== undefined) {
      const mismatch = energy.value * 10 - context.arcEnergy;
      if (Math.abs(mismatch) > 3 && context.trackElapsed > 60) {
        if (mismatch > 0) candidate = { cat: 'regulate', int: 'subtle', reason: 'energy_above_arc' };
        else if (!isWindDown) candidate = { cat: 'stimulate', int: 'subtle', reason: 'energy_below_arc' };
      }
    }

    if (!candidate && trajectory.direction === 'volatile' && trajectory.confidence > 0.3)
      candidate = { cat: 'transition', int: 'subtle', reason: 'volatile_stabilize' };

    if (!candidate && energy.level === 'low' && isActive && immersion.level !== 'engaged')
      candidate = { cat: 'stimulate', int: 'moderate', reason: 'low_energy_active_intent' };

    if (!candidate && immersion.level === 'engaged' && trajectory.direction === 'improving' && context.trackElapsed > 200)
      candidate = { cat: 'hold', int: 'subtle', reason: 'high_engagement_extend' };

    if (!candidate && immersion.level === 'detached' && context.trackElapsed > 90) {
      this.observationCount++;
      if (this.observationCount > 5) candidate = { cat: 'stimulate', int: 'moderate', reason: 'detached_persistent' };
    }

    // === CIRCUIT BREAKER: Block reasons that have failed 3+ times (with recovery) ===
    // V4: Recovery — unblock reasons after RECOVERY_TIME (10 min) of silence
    if (this.blockedReasons.size > 0) {
      for (const reason of [...this.blockedReasons]) {
        const blockedTime = this.blockedAt[reason] || 0;
        if (now - blockedTime > this.RECOVERY_TIME) {
          this.blockedReasons.delete(reason);
          delete this.blockedAt[reason];
          this.reasonFailures[reason] = { bad: 0 }; // Reset strikes
        }
      }
    }

    if (candidate && this.blockedReasons.has(candidate.reason)) {
      this._emitNearDecision(candidate, 'blocked_by_circuit_breaker', state);
      return null;
    }

    // === FIRE or DO NOTHING ===
    if (candidate) return this._makeDecision(candidate.cat, candidate.int, candidate.reason, state);
    return null;
  }

  // Phase 4b: Log near-decisions (interventions considered but not fired)
  _emitNearDecision(candidate, blockReason, state) {
    // Stored for offline causal analysis — these are the control group
    if (this._nearDecisionLog) {
      this._nearDecisionLog.push({
        candidate, blockReason, state: { energy: state.energy.level, immersion: state.immersion.level },
        timestamp: Date.now()
      });
    }
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
    this.interventionTimestamps.push(Date.now()); // Phase 1b: hourly cap tracking
    this.recentActions.push(decision);
    if (this.recentActions.length > this.MAX_RECENT) this.recentActions.shift();
    if (this.feedback) this.feedback.registerIntervention(decision, state);
    return decision;
  }

  onFeedback(outcome, reason) {
    // Global consecutive failure tracking
    if (outcome === 'negative') { this.consecutiveFailures = Math.min(4, this.consecutiveFailures + 1.5); }
    else if (outcome === 'positive') { this.consecutiveFailures = Math.max(0, this.consecutiveFailures - 2); }
    else { this.consecutiveFailures = Math.min(4, this.consecutiveFailures + 0.3); }
    this.lastFailureDecayTime = Date.now();

    // Phase 1a: Per-reason circuit breaker
    if (reason) {
      if (!this.reasonFailures[reason]) this.reasonFailures[reason] = { bad: 0 };
      if (outcome === 'positive') {
        this.reasonFailures[reason].bad = Math.max(0, this.reasonFailures[reason].bad - 1);
      } else {
        // Neutral and negative both count as strikes (L86: neutral = mild failure)
        this.reasonFailures[reason].bad++;
        if (this.reasonFailures[reason].bad >= this.REASON_STRIKE_LIMIT) {
          this.blockedReasons.add(reason);
          this.blockedAt[reason] = Date.now(); // V4: timestamp for recovery
        }
      }
    }

    // Phase 1b: Consecutive neutral/negative → 15-min suppression
    if (outcome === 'positive') {
      this.consecutiveNeutralNegative = 0;
    } else {
      this.consecutiveNeutralNegative++;
      if (this.consecutiveNeutralNegative >= 3) {
        this.suppressionUntil = Date.now() + 900000; // 15 minutes
        this.consecutiveNeutralNegative = 0;
      }
    }
  }

  toJSON() {
    return {
      cooldown: this.baseCooldown * Math.pow(1.3, this.consecutiveFailures),
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

    // V4: Artist variety tracking — avoid consecutive same-artist plays
    this.recentArtists = [];
    this.MAX_RECENT_ARTISTS = 4;
    // V4: All selected intents (not just first)
    this.allIntents = [];
    this.allVibes = [];
  }

  initialize(context) {
    // V4: Store all intents/vibes for blended query generation
    this.allIntents = context.intent || [];
    this.allVibes = context.vibe || [];
    const primaryIntent = this.allIntents[0] || 'unwind';
    const sleepHours = parseFloat(context.sleep_in_hours) || 3;
    this.currentProfile = this.sonicProfiles[primaryIntent] || this.sonicProfiles.unwind;
    this.currentArc = this._designArc(primaryIntent, sleepHours);
    return { profile: primaryIntent, arcPoints: this.currentArc.length, sessionDuration: this.currentArc[this.currentArc.length - 1]?.t || 30 };
  }

  buildInitialQueries(context) {
    const intents = context.intent || [];
    const vibes = context.vibe || [];
    const primaryIntent = intents[0] || 'unwind';
    const primaryVibe = vibes[0] || 'mixed';
    const profile = this.sonicProfiles[primaryIntent] || this.sonicProfiles.unwind;
    const queries = [];
    const phases = this._getArcPhases();

    for (const phase of phases) {
      // V4: Primary query from primary intent+vibe
      const primaryVibeQueries = profile.vibes[primaryVibe] || profile.vibes.mixed;
      const baseQuery = primaryVibeQueries[Math.floor(Math.random() * primaryVibeQueries.length)];
      queries.push({
        query: baseQuery, phase: phase.name, energy: phase.energy, count: phase.trackCount,
        duration: profile.trackDuration.min + Math.random() * (profile.trackDuration.max - profile.trackDuration.min)
      });

      // V4: Blend — add 1 query from a secondary intent if available
      if (intents.length > 1) {
        const secIntent = intents[1 + Math.floor(Math.random() * (intents.length - 1))];
        const secProfile = this.sonicProfiles[secIntent];
        if (secProfile) {
          const secVibe = vibes.length > 1 ? vibes[Math.floor(Math.random() * vibes.length)] : primaryVibe;
          const secVibeQueries = secProfile.vibes[secVibe] || secProfile.vibes.mixed;
          queries.push({
            query: secVibeQueries[Math.floor(Math.random() * secVibeQueries.length)],
            phase: phase.name, energy: Math.round((phase.energy + secProfile.energyRange[0]) / 2),
            count: 2, duration: secProfile.trackDuration.min
          });
        }
      }
    }
    return queries;
  }

  getNextSearchQuery(state, sessionElapsed) {
    if (!this.currentProfile || !this.currentArc) return null;
    const targetEnergy = this.getTargetEnergy(sessionElapsed);
    const energyDesc = this._energyToDescriptor(targetEnergy);
    const stateMod = this._stateToModifier(state);

    // V4: Blend across all selected intents, not just primary
    let pool = this.currentProfile;
    if (this.allIntents.length > 1 && Math.random() < 0.35) {
      // 35% chance: pull from a secondary intent's profile for variety
      const secIntent = this.allIntents[1 + Math.floor(Math.random() * (this.allIntents.length - 1))];
      pool = this.sonicProfiles[secIntent] || this.currentProfile;
    }
    const vibeKeys = Object.keys(pool.vibes);
    const randomVibe = pool.vibes[vibeKeys[Math.floor(Math.random() * vibeKeys.length)]];
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

  // V4: Track artist and deprioritize repeats when ordering results
  trackArtist(artist) {
    if (!artist) return;
    this.recentArtists.push(artist.toLowerCase());
    if (this.recentArtists.length > this.MAX_RECENT_ARTISTS) this.recentArtists.shift();
  }

  sortByArtistVariety(tracks) {
    if (this.recentArtists.length === 0) return tracks;
    // Partition: new artists first, then recent artists
    const fresh = tracks.filter(t => !this.recentArtists.includes((t.artist || '').toLowerCase()));
    const repeat = tracks.filter(t => this.recentArtists.includes((t.artist || '').toLowerCase()));
    return [...fresh, ...repeat];
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

  // adaptArc() KILLED — Session 59 Phase 3a.
  // The arc is the designed experience. Immutable. System closes gaps via interventions, not by redrawing the map.
  // DJ override (D-012) is the only valid arc mutation source — not yet implemented.

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
// FACE ANALYZER — face-api.js (68 landmarks, EAR-based eye detection)
// Interface: analyze(detection) → { eyes, headPose, nod, brow, mouth, engagement }
// Compatible with StateEstimator.addFaceData()
// ============================================================

class FaceAnalyzer {
  constructor() {
    this.headPoseHistory = [];
    this.MAX_POSE_HISTORY = 10;
  }

  // Takes face-api.js detection result, returns engine-compatible face data
  analyze(detection) {
    if (!detection || !detection.landmarks) {
      return { eyes: 'Unknown', headPose: 'Unknown', nod: 'None', brow: 'Normal', mouth: 'Closed', engagement: 30 };
    }

    const positions = detection.landmarks.positions;
    const expressions = detection.expressions || {};

    const eyes = this._analyzeEyes(positions);
    const headPose = this._analyzeHeadPose(positions);
    const nod = this._analyzeNod(headPose);
    const brow = this._analyzeBrow(positions);
    const mouth = this._analyzeMouth(positions);
    const engagement = this._computeEngagement(eyes, headPose, mouth, expressions);

    return { eyes, headPose: headPose.pose, nod, brow, mouth, engagement };
  }

  _analyzeEyes(positions) {
    // Eye Aspect Ratio (EAR) using face-api.js 68-point landmarks
    // Left eye: points 36-41, Right eye: points 42-47
    const leftEAR = this._computeEAR(positions, [36, 37, 38, 39, 40, 41]);
    const rightEAR = this._computeEAR(positions, [42, 43, 44, 45, 46, 47]);
    const avgEAR = (leftEAR + rightEAR) / 2;

    if (avgEAR < 0.18) return 'Closed';
    if (avgEAR < 0.24) return 'Droopy';
    return 'Open';
  }

  _computeEAR(positions, indices) {
    // EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    const p = indices.map(i => positions[i]);
    if (!p[0] || !p[3]) return 0.3; // default open
    const vertical1 = Math.hypot(p[1].x - p[5].x, p[1].y - p[5].y);
    const vertical2 = Math.hypot(p[2].x - p[4].x, p[2].y - p[4].y);
    const horizontal = Math.hypot(p[0].x - p[3].x, p[0].y - p[3].y);
    return horizontal > 0 ? (vertical1 + vertical2) / (2 * horizontal) : 0.3;
  }

  _analyzeHeadPose(positions) {
    // Crude head pose from nose tip (30) relative to face center
    const noseTip = positions[30];
    const leftCheek = positions[0];
    const rightCheek = positions[16];
    const chin = positions[8];
    const forehead = positions[27]; // bridge of nose top as proxy

    let pitch = 0, yaw = 0;
    if (noseTip && leftCheek && rightCheek && chin && forehead) {
      const faceCenterX = (leftCheek.x + rightCheek.x) / 2;
      const faceCenterY = (forehead.y + chin.y) / 2;
      const faceWidth = Math.abs(rightCheek.x - leftCheek.x) || 1;
      const faceHeight = Math.abs(chin.y - forehead.y) || 1;

      yaw = (noseTip.x - faceCenterX) / faceWidth;
      pitch = (noseTip.y - faceCenterY) / faceHeight;
    }

    let pose;
    if (pitch > 0.15) pose = 'Down';
    else if (pitch < -0.1) pose = 'Up';
    else pose = 'Forward';

    return { pose, pitch, yaw };
  }

  _analyzeNod(headPose) {
    this.headPoseHistory.push({ pitch: headPose.pitch, yaw: headPose.yaw, t: Date.now() });
    if (this.headPoseHistory.length > this.MAX_POSE_HISTORY) this.headPoseHistory.shift();

    if (this.headPoseHistory.length < 3) return 'None';

    const pitches = this.headPoseHistory.map(h => h.pitch);
    const mean = pitches.reduce((a, b) => a + b, 0) / pitches.length;
    const variance = pitches.reduce((s, v) => s + (v - mean) ** 2, 0) / pitches.length;

    if (variance > 0.02) return 'Active';
    if (variance > 0.005) return 'Light';
    return 'None';
  }

  _analyzeBrow(positions) {
    // Brow raise: distance from brow points (17-21 left, 22-26 right) to eye top
    const browL = positions[19]; // left brow center
    const eyeL = positions[37]; // left eye top
    const browR = positions[24]; // right brow center
    const eyeR = positions[44]; // right eye top

    if (!browL || !eyeL || !browR || !eyeR) return 'Normal';

    const leftDist = Math.abs(browL.y - eyeL.y);
    const rightDist = Math.abs(browR.y - eyeR.y);
    const avgDist = (leftDist + rightDist) / 2;

    // Normalized by face height
    const chin = positions[8];
    const forehead = positions[27];
    const faceHeight = Math.abs(chin.y - forehead.y) || 1;
    const normalized = avgDist / faceHeight;

    return normalized > 0.18 ? 'Raised' : 'Normal';
  }

  _analyzeMouth(positions) {
    // Mouth open: distance from top lip (62) to bottom lip (66)
    const topLip = positions[62];
    const bottomLip = positions[66];
    if (!topLip || !bottomLip) return 'Closed';

    const mouthOpen = Math.abs(bottomLip.y - topLip.y);
    const chin = positions[8];
    const forehead = positions[27];
    const faceHeight = Math.abs(chin.y - forehead.y) || 1;
    const normalized = mouthOpen / faceHeight;

    return normalized > 0.06 ? 'Open' : 'Closed';
  }

  _computeEngagement(eyes, headPose, mouth, expressions) {
    // D-046: No facial expression classification (emotion labels killed — Barrett 2019).
    // Engagement from structural features only.

    let score = 50;

    // Eye openness
    if (eyes === 'Open') score += 10;
    else if (eyes === 'Droopy') score -= 8;
    else score -= 18; // Closed

    // Head pose
    if (headPose.pose === 'Forward') score += 6;
    else if (headPose.pose === 'Down') score -= 10;
    else score -= 4; // Up

    // Mouth reaction
    if (mouth === 'Open') score += 6;

    // Head movement — subtle = engaged (nodding to music, swaying)
    if (this.headPoseHistory.length >= 3) {
      const pitches = this.headPoseHistory.slice(-5).map(h => h.pitch);
      const mean = pitches.reduce((a, b) => a + b, 0) / pitches.length;
      const variance = pitches.reduce((s, v) => s + (v - mean) ** 2, 0) / pitches.length;
      if (variance > 0.005 && variance < 0.03) score += 8; // subtle rhythmic movement
      else if (variance >= 0.03) score -= 3; // too fidgety
    }

    // Use face-api.js expression scores as mild engagement signal (not classification)
    // Higher neutral = less engaged; higher surprise/happy = more engaged
    const neutral = expressions.neutral || 0;
    const surprise = expressions.surprised || 0;
    const happy = expressions.happy || 0;
    score += (surprise + happy) * 10;
    score -= neutral > 0.8 ? 5 : 0;

    return Math.max(0, Math.min(100, Math.round(score)));
  }
}

// ============================================================
// GUEST MODELER — Per-session learning + cross-event skeleton (Phase 4a)
// Bifurcated: trait layer (stable across events) + state layer (this session)
// ============================================================

class GuestModeler {
  constructor() {
    // State layer — this session only, fast-learning
    this.sessionModel = {
      interventionResponses: {},   // { reason: { positive: N, neutral: N, negative: N } }
      nudgeTolerance: 'MEDIUM',    // LOW / MEDIUM / HIGH — from intake
      intentVector: null,          // { energy, calmness } from intake
      initialEnergy: 0.5,         // V4: Self-reported energy level (1-5 → 0-1)
      calibrationBaseline: null,   // Snapshot from Calibrator
      peakMoments: [],             // Timestamps of detected peaks
    };

    // Trait layer — placeholder for cross-event persistence
    this.traitLayer = {
      arousalPreference: 0.5,
      nudgeReceptivity: 0.5,
      socialOpenness: 0.5,
    };
  }

  initFromIntake(context) {
    const intent = context.intent || [];

    // V4: Direct intake wiring — openness feeds nudge tolerance directly
    if (context.openness) {
      const map = { low: 'LOW', medium: 'MEDIUM', high: 'HIGH' };
      this.sessionModel.nudgeTolerance = map[context.openness] || 'MEDIUM';
    } else {
      // Fallback: infer from intent (V3 behavior)
      if (intent.includes('energy') || intent.includes('focus')) this.sessionModel.nudgeTolerance = 'HIGH';
      else if (intent.includes('escape') || intent.includes('sleep_prep')) this.sessionModel.nudgeTolerance = 'LOW';
    }

    // V4: Direct energy level from intake slider (1-5 → 0-1)
    if (context.energy_level) {
      this.sessionModel.initialEnergy = (parseInt(context.energy_level) - 1) / 4;
    }

    // V4: Use "before" context to adjust initial energy estimate
    const before = context.before || [];
    let energyMod = 0;
    if (before.includes('exercise')) energyMod += 0.15;
    if (before.includes('work')) energyMod += 0.05;
    if (before.includes('nothing')) energyMod -= 0.1;
    if (before.includes('social')) energyMod += 0.1;
    this.sessionModel.initialEnergy = Math.max(0, Math.min(1, this.sessionModel.initialEnergy + energyMod));

    // Intent vector from actual selections
    this.sessionModel.intentVector = {
      energy: intent.includes('energy') ? 0.8 : intent.includes('sleep_prep') ? 0.2 : 0.5,
      calmness: intent.includes('unwind') || intent.includes('sleep_prep') ? 0.8 : 0.3,
    };

    // V4: Set initial nudge receptivity from openness
    if (context.openness === 'high') this.traitLayer.nudgeReceptivity = 0.7;
    else if (context.openness === 'low') this.traitLayer.nudgeReceptivity = 0.3;
  }

  // Record intervention outcome for this session's learning
  recordOutcome(reason, outcome) {
    if (!this.sessionModel.interventionResponses[reason]) {
      this.sessionModel.interventionResponses[reason] = { positive: 0, neutral: 0, negative: 0 };
    }
    this.sessionModel.interventionResponses[reason][outcome]++;

    // Update nudge receptivity based on outcomes
    if (outcome === 'positive') this.traitLayer.nudgeReceptivity = Math.min(1, this.traitLayer.nudgeReceptivity + 0.05);
    else if (outcome === 'negative') this.traitLayer.nudgeReceptivity = Math.max(0, this.traitLayer.nudgeReceptivity - 0.1);
  }

  // Get nudge tolerance threshold modifier for InterventionEngine
  getNudgeThreshold() {
    const toleranceMap = { LOW: 1, MEDIUM: 3, HIGH: 5 };
    return toleranceMap[this.sessionModel.nudgeTolerance] || 3;
  }

  toJSON() {
    return {
      nudgeTolerance: this.sessionModel.nudgeTolerance,
      initialEnergy: this.sessionModel.initialEnergy,
      nudgeReceptivity: Math.round(this.traitLayer.nudgeReceptivity * 100) / 100,
      interventionResponses: this.sessionModel.interventionResponses
    };
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
    this.guestModeler = new GuestModeler();

    // Phase 4b: Near-decision log — interventions considered but not fired (control group)
    this.intervention._nearDecisionLog = [];

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
    this.guestModeler.initFromIntake(context);
    // Phase 4c: Wire nudge tolerance to intervention engine hourly cap
    this.intervention.MAX_PER_HOUR = this.guestModeler.getNudgeThreshold();

    // V4: Seed Kalman filters from intake energy level (warm start instead of cold 0.5)
    const initialEnergy = this.guestModeler.sessionModel.initialEnergy;
    if (initialEnergy !== undefined) {
      this.stateEstimator.kf.energy.x = initialEnergy;
      this.stateEstimator.kf.energy.P = 0.5; // moderate uncertainty — we trust intake somewhat
    }

    this.loopId = setInterval(() => this._tick(), 5000);

    this._emit('log', {
      type: 'engine_start',
      message: `Engine started — calibrating for ${this.calibrator.calibrationDuration / 1000}s`,
      context: { intent: context.intent, vibe: context.vibe, sleepHours: context.sleep_in_hours, initialEnergy }
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
    // V4: If a previous track was playing, log natural completion as music interaction
    if (this.currentTrack && this.trackStartedAt) {
      const elapsed = (Date.now() - this.trackStartedAt) / 1000;
      if (elapsed > 60) { // Only count as completion if played >60s
        this.stateEstimator.addMusicInteraction('complete', elapsed);
      }
    }
    this.currentTrack = track;
    this.trackStartedAt = Date.now();
    this._pendingSeek = true; // trigger seek evaluation on next tick
  }

  // V4: Log user skip as music interaction
  onTrackSkipped() {
    const elapsed = this.trackStartedAt ? (Date.now() - this.trackStartedAt) / 1000 : 0;
    this.stateEstimator.addMusicInteraction('skip', elapsed);
  }

  // V4: Log moment button press as music interaction
  onMomentPress() {
    const elapsed = this.trackStartedAt ? (Date.now() - this.trackStartedAt) / 1000 : 0;
    this.stateEstimator.addMusicInteraction('moment', elapsed);
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

    // V4: Pass arc target to StateEstimator for sparse-signal drift (W-4)
    this.stateEstimator._arcTarget = this.musicBrain.getTargetEnergy(sessionElapsed);
    const state = this.stateEstimator.estimate(this.context);

    this._emit('state', {
      type: 'state_update',
      state: { energy: state.energy, immersion: state.immersion, trajectory: state.trajectory },
      raw: state.raw,
      calibration: { calibrated: this.calibrator.calibrated, progress: this.calibrator.getProgress() },
      arc: { target: Math.round(this.musicBrain.getTargetEnergy(sessionElapsed) * 10) / 10, sessionMinutes: Math.round(sessionElapsed * 10) / 10 },
      cooldown: {
        remaining: Math.max(0, Math.round((this.intervention.lastInterventionTime + this.intervention.baseCooldown * Math.pow(1.3, this.intervention.consecutiveFailures) - now) / 1000)),
        failures: this.intervention.consecutiveFailures
      },
      timestamp: now
    });

    const feedbackOutcomes = this.feedback.tick(state);
    for (const outcome of feedbackOutcomes) {
      this.intervention.onFeedback(outcome.outcome.outcome, outcome.intervention.reason);
      this.guestModeler.recordOutcome(outcome.intervention.reason, outcome.outcome.outcome);
      this._emit('log', { type: 'feedback', message: `"${outcome.intervention.reason}" scored: ${outcome.outcome.outcome}`, outcome });
    }

    const trackElapsed = this.trackStartedAt ? (now - this.trackStartedAt) / 1000 : 0;
    const decision = this.intervention.decide(state, {
      intent: this.context?.intent || [], calibrated: this.calibrator.calibrated,
      trackElapsed, arcEnergy: this.musicBrain.getTargetEnergy(sessionElapsed), sessionElapsed
    }, sessionElapsed);

    if (decision) this._executeDecision(decision, state, sessionElapsed);
    // adaptArc KILLED (Phase 3a) — arc is immutable. System closes gap via interventions, not by redrawing the map.
    // _evaluateMusic KILLED (Phase 1c) — one decision point, one feedback loop. Music changes go through InterventionEngine only.

    // V4: Autoplay reassertion — if engine hasn't searched in 5+ min, proactively search
    // This prevents YouTube queue from driving the entire session when circuit breaker is active
    if (!this._lastEngineSearchTime) this._lastEngineSearchTime = now;
    if (decision && decision.action && decision.action.includes('music')) this._lastEngineSearchTime = now;
    const silenceMin = (now - this._lastEngineSearchTime) / 60000;
    if (silenceMin > 5 && this.calibrator.calibrated) {
      const query = this.musicBrain.getNextSearchQuery(state, sessionElapsed);
      if (query) {
        this._emit('command', { type: 'music', command: 'queue_next', query });
        this._lastEngineSearchTime = now;
        this._emit('log', { type: 'autoplay_reassert', message: `Engine silent ${Math.round(silenceMin)}m — proactive search: ${query.query}` });
      }
    }

    // V4: Seek jumping KILLED — always play tracks from the beginning.
    // Jumping to 35%/70% was disorienting on phone. Let the music breathe.
    if (this._pendingSeek) this._pendingSeek = false;

    this._emit('log', {
      type: 'tick',
      state: { energy: state.energy.level, immersion: state.immersion.level, trajectory: state.trajectory.direction },
      engagement: state.raw.avgEngagement30s,
      arcTarget: Math.round(this.musicBrain.getTargetEnergy(sessionElapsed) * 10) / 10,
      interventionPending: this.feedback.pending.length,
      cooldown: this.intervention.toJSON().cooldown
    });
  }

  // _evaluateMusic() KILLED — Session 59 Phase 1c.
  // One decision point (InterventionEngine), one feedback loop. No parallel music actuator.

  // Estimate where in a track to start based on target energy
  _getSeekPosition(trackDuration, targetEnergy) {
    if (!trackDuration || trackDuration < 90) return 0;
    // Rough energy map: 0-15% intro, 15-40% build, 40-75% peak, 75-100% outro
    if (targetEnergy >= 7) return Math.floor(trackDuration * 0.35); // jump to peak buildup
    if (targetEnergy >= 5) return Math.floor(trackDuration * 0.15); // start at build
    if (targetEnergy >= 3) return 0; // play from start (mellow is fine from intro)
    return Math.floor(trackDuration * 0.70); // low energy — play fade-out section
  }

  _executeDecision(decision, state, sessionElapsed) {
    this._emit('log', { type: 'intervention_fired', message: `[${decision.category}/${decision.intensity}] ${decision.reason}`, decision });
    this._emit('intervention', decision);

    if (decision.text) this._emit('command', { type: 'nudge', text: decision.text, decision });
    if (decision.haptic) this._emit('command', { type: 'haptic', pattern: decision.haptic });
    if (decision.action) {
      let action = decision.action;

      // V4: Minimum play enforcement — no skip if track played < 90s
      const trackElapsed = this.trackStartedAt ? (Date.now() - this.trackStartedAt) / 1000 : 0;
      if (action === 'skip_track' && trackElapsed < 90) {
        // Downgrade: queue next track instead of interrupting current
        action = (decision.category === 'stimulate') ? 'music_shift_up' : 'music_shift_down';
        this._emit('log', { type: 'min_play_guard', message: `Skip blocked (${Math.round(trackElapsed)}s < 90s) — downgraded to ${action}` });
      }

      // Phase 3c: Congruence gate — ensure music action direction matches intervention category
      const congruent = this._checkCongruence(decision.category, action);
      if (congruent) {
        const musicCommand = this.musicBrain.interpretAction(action, state, sessionElapsed);
        if (musicCommand) this._emit('command', { type: 'music', ...musicCommand });
      } else {
        this._emit('log', { type: 'congruence_block', message: `Music action "${action}" blocked — incongruent with "${decision.category}"` });
      }
    }
  }

  // Phase 3c: Congruence gate — arousal direction alignment between intervention and music
  _checkCongruence(category, action) {
    // Map: which music actions are congruent with which intervention categories
    const congruenceMap = {
      stimulate: ['music_shift_up', 'volume_up', 'skip_track'],           // Energizing actions
      regulate:  ['music_shift_down', 'volume_down', 'skip_track'],       // Calming actions
      transition: ['music_arc_align', 'skip_track', 'music_shift_up', 'music_shift_down'], // Either direction OK
      hold:      ['extend_track']                                          // Only extend
    };
    const allowed = congruenceMap[category];
    if (!allowed) return true; // Unknown category — pass through
    return allowed.includes(action);
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
      guestModel: this.guestModeler.toJSON(),
      nearDecisions: (this.intervention._nearDecisionLog || []).length,
      blockedReasons: Array.from(this.intervention.blockedReasons),
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
// ANALYTICS LOGGER — V4 Structured data capture for analysis
// Replaces V3 SessionLogger. Typed event streams, auto-sync to GitHub.
// Data format designed for pandas/analysis in Claude Code sessions.
// ============================================================

class AnalyticsLogger {
  constructor(options = {}) {
    this.githubRepo = options.githubRepo || null;
    this.githubToken = options.githubToken || null;
    this._githubSaving = false;
    this._fileSha = null;
    this._autoSaveId = null;
    this._githubSaveId = null;
    this._sensorBuffer = [];
    this._sessionStart = null;

    this.data = {
      version: '4.0',
      session_id: null,
      meta: {},
      timeline: [],      // 5s engine ticks — main time series
      interventions: [],  // each intervention with outcome attached
      music: [],          // play/skip/search/queue events
      face: [],           // every face detection result
      sensors: [],        // downsampled to 5s averages
      moments: [],        // user-tapped "I feel something"
      nudge_feedback: [], // user thumbs up/down on nudges
      summary: null       // computed at session end
    };
  }

  startSession(sessionId, intake) {
    this._sessionStart = Date.now();
    this.data.session_id = sessionId;
    this.data.meta = {
      started_at: new Date().toISOString(),
      ended_at: null,
      duration_min: 0,
      intake,
      device: {
        userAgent: navigator.userAgent,
        screen: `${screen.width}x${screen.height}`,
        platform: navigator.platform,
        touchPoints: navigator.maxTouchPoints || 0
      },
      engine_version: '4.0'
    };
    this._autoSaveId = setInterval(() => this._saveLocal(), 30000);
    if (this.githubRepo && this.githubToken) {
      this._githubSaveId = setInterval(() => this._saveGitHub(), 120000);
    }
  }

  _elapsed() { return this._sessionStart ? Math.round((Date.now() - this._sessionStart) / 1000) : 0; }

  // --- Typed log methods ---

  logTick(stateData) {
    // stateData comes from engine 'state' event (type: state_update)
    const st = stateData.state;
    this.data.timeline.push({
      t: Date.now(), s: this._elapsed(),
      e_val: st.energy.value, e_lvl: st.energy.level, e_conf: st.energy.confidence,
      i_val: st.immersion.value, i_lvl: st.immersion.level, i_conf: st.immersion.confidence,
      traj: st.trajectory.direction, traj_val: st.trajectory.value,
      arc: stateData.arc?.target || null,
      cd: stateData.cooldown?.remaining || 0,
      cal: stateData.calibration?.calibrated ? 1 : (stateData.calibration?.progress || 0)
    });
  }

  logIntervention(decision) {
    this.data.interventions.push({
      t: decision.timestamp || Date.now(), s: this._elapsed(),
      cat: decision.category, int: decision.intensity, reason: decision.reason,
      text: decision.text || null, action: decision.action || null, haptic: decision.haptic || null,
      state: decision.state || null,
      outcome: null, score: null, factors: null // filled when feedback arrives
    });
  }

  logFeedbackOutcome(outcome) {
    // Match to the intervention by timestamp
    const intRecord = this.data.interventions.find(i => i.t === outcome.timestamp);
    if (intRecord) {
      intRecord.outcome = outcome.outcome.outcome;
      intRecord.score = outcome.outcome.score;
      intRecord.factors = outcome.outcome.factors;
    }
  }

  logMusic(eventType, track, extra = {}) {
    this.data.music.push({
      t: Date.now(), s: this._elapsed(),
      event: eventType,
      track_id: track?.id || null, title: track?.title || null, artist: track?.artist || null,
      query: extra.query || null, reason: extra.reason || null
    });
  }

  logFace(faceData) {
    this.data.face.push({
      t: Date.now(), s: this._elapsed(),
      eyes: faceData.eyes, head: faceData.headPose, nod: faceData.nod,
      brow: faceData.brow, mouth: faceData.mouth,
      eng: faceData.engagement, conf: faceData.detectionConfidence || null
    });
  }

  logSensor(sensorData) {
    this._sensorBuffer.push(sensorData);
    // Flush every 5 readings → 1 averaged entry (5s at 1s intervals)
    if (this._sensorBuffer.length >= 5) {
      const avg = (key) => {
        const vals = this._sensorBuffer.map(s => s[key]).filter(v => v !== undefined);
        return vals.length > 0 ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length * 100) / 100 : 0;
      };
      this.data.sensors.push({
        t: Date.now(), s: this._elapsed(),
        ax: avg('ax'), ay: avg('ay'), az: avg('az'),
        gx: avg('gx'), gy: avg('gy'), gz: avg('gz')
      });
      this._sensorBuffer = [];
    }
  }

  logMoment(state, track) {
    this.data.moments.push({
      t: Date.now(), s: this._elapsed(),
      state: { e: state.energy?.level, i: state.immersion?.level, traj: state.trajectory?.direction },
      track: track?.title || null
    });
  }

  logNudgeFeedback(interventionTimestamp, response) {
    this.data.nudge_feedback.push({
      t: Date.now(), s: this._elapsed(),
      intervention_t: interventionTimestamp, response
    });
    // Also update the intervention record
    const intRecord = this.data.interventions.find(i => i.t === interventionTimestamp);
    if (intRecord) intRecord.user_feedback = response;
  }

  logEvent(type, data) {
    // Generic fallback for events that don't fit typed streams
    if (!this.data._misc) this.data._misc = [];
    this.data._misc.push({ t: Date.now(), s: this._elapsed(), type, ...data });
  }

  // --- Session end: compute summary ---

  async endSession(engineSummary) {
    this.data.meta.ended_at = new Date().toISOString();
    this.data.meta.duration_min = engineSummary?.duration || Math.round(this._elapsed() / 60);

    // Compute analytics summary
    const tl = this.data.timeline;
    this.data.summary = {
      duration_min: this.data.meta.duration_min,
      tracks_played: this.data.music.filter(m => m.event === 'play').length,
      tracks_skipped_user: this.data.music.filter(m => m.reason === 'user_skip').length,
      tracks_skipped_engine: this.data.music.filter(m => m.reason === 'engine_skip').length,
      interventions_total: this.data.interventions.length,
      interventions_by_cat: {},
      interventions_positive: this.data.interventions.filter(i => i.outcome === 'positive').length,
      interventions_neutral: this.data.interventions.filter(i => i.outcome === 'neutral').length,
      interventions_negative: this.data.interventions.filter(i => i.outcome === 'negative').length,
      blocked_reasons: engineSummary?.blockedReasons || [],
      near_decisions: engineSummary?.nearDecisions || 0,
      nudge_tolerance: engineSummary?.guestModel?.nudgeTolerance || 'MEDIUM',
      nudge_receptivity: engineSummary?.guestModel?.nudgeReceptivity || 0.5,
      calibration: engineSummary?.calibration || 'incomplete',
      face_readings: this.data.face.length,
      sensor_readings: this.data.sensors.length,
      moments: this.data.moments.length,
      user_feedback_count: this.data.nudge_feedback.length,
      avg_energy: tl.length > 0 ? Math.round(tl.reduce((s, t) => s + t.e_val, 0) / tl.length * 100) / 100 : null,
      avg_immersion: tl.length > 0 ? Math.round(tl.reduce((s, t) => s + t.i_val, 0) / tl.length * 100) / 100 : null,
      energy_range: tl.length > 0 ? [
        Math.round(Math.min(...tl.map(t => t.e_val)) * 100) / 100,
        Math.round(Math.max(...tl.map(t => t.e_val)) * 100) / 100
      ] : null,
      immersion_range: tl.length > 0 ? [
        Math.round(Math.min(...tl.map(t => t.i_val)) * 100) / 100,
        Math.round(Math.max(...tl.map(t => t.i_val)) * 100) / 100
      ] : null
    };
    // Intervention breakdown by category
    for (const int of this.data.interventions) {
      this.data.summary.interventions_by_cat[int.cat] = (this.data.summary.interventions_by_cat[int.cat] || 0) + 1;
    }

    // Final saves
    if (this._autoSaveId) { clearInterval(this._autoSaveId); this._autoSaveId = null; }
    if (this._githubSaveId) { clearInterval(this._githubSaveId); this._githubSaveId = null; }
    this._saveLocal();
    if (this.githubRepo && this.githubToken) {
      await this._saveGitHub();
    }
    return this.data;
  }

  // --- Persistence ---

  _saveLocal() {
    try {
      localStorage.setItem(`me_v4_${this.data.session_id}`, JSON.stringify(this.data));
      const index = JSON.parse(localStorage.getItem('me_v4_sessions') || '[]');
      if (!index.includes(this.data.session_id)) {
        index.push(this.data.session_id);
        localStorage.setItem('me_v4_sessions', JSON.stringify(index));
      }
    } catch (e) { /* storage full — oldest sessions could be pruned */ }
  }

  async _saveGitHub() {
    if (!this.githubRepo || !this.githubToken || this._githubSaving) return false;
    this._githubSaving = true;
    try {
      const filename = `data/sessions/${this.data.session_id}.json`;
      const content = btoa(unescape(encodeURIComponent(JSON.stringify(this.data))));
      const url = `https://api.github.com/repos/${this.githubRepo}/contents/${filename}`;
      const headers = { 'Authorization': `token ${this.githubToken}`, 'Content-Type': 'application/json' };

      let sha = this._fileSha;
      if (!sha) {
        try {
          const existing = await fetch(url, { headers });
          if (existing.ok) sha = (await existing.json()).sha;
        } catch (e) {}
      }

      const body = { message: `V4 ${this.data.session_id} — ${this.data.timeline.length} ticks, ${this.data.interventions.length} interventions`, content };
      if (sha) body.sha = sha;

      const resp = await fetch(url, { method: 'PUT', headers, body: JSON.stringify(body) });
      if (resp.ok) {
        this._fileSha = (await resp.json()).content.sha;
        this._githubSaving = false;
        return true;
      }
    } catch (e) { console.warn('GitHub save failed:', e); }
    this._githubSaving = false;
    return false;
  }

  // --- Static utilities ---

  static async syncPending(githubRepo, githubToken) {
    if (!githubRepo || !githubToken) return { synced: 0, failed: 0 };
    const index = JSON.parse(localStorage.getItem('me_v4_sessions') || '[]');
    let synced = 0, failed = 0;
    for (const id of index) {
      const raw = localStorage.getItem(`me_v4_${id}`);
      if (!raw) continue;
      const data = JSON.parse(raw);
      if (!data.summary) continue; // incomplete — skip
      try {
        const filename = `data/sessions/${id}.json`;
        const content = btoa(unescape(encodeURIComponent(raw)));
        const url = `https://api.github.com/repos/${githubRepo}/contents/${filename}`;
        const headers = { 'Authorization': `token ${githubToken}`, 'Content-Type': 'application/json' };
        let sha = null;
        try { const ex = await fetch(url, { headers }); if (ex.ok) sha = (await ex.json()).sha; } catch (e) {}
        const body = { message: `V4 ${id} (sync)`, content };
        if (sha) body.sha = sha;
        const resp = await fetch(url, { method: 'PUT', headers, body: JSON.stringify(body) });
        if (resp.ok) synced++; else failed++;
      } catch (e) { failed++; }
    }
    return { synced, failed };
  }

  static getAllSessions() {
    const index = JSON.parse(localStorage.getItem('me_v4_sessions') || '[]');
    return index.map(id => JSON.parse(localStorage.getItem(`me_v4_${id}`) || 'null')).filter(Boolean);
  }

  static exportAll() {
    const sessions = AnalyticsLogger.getAllSessions();
    const blob = new Blob([JSON.stringify(sessions, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `me-v4-${new Date().toISOString().slice(0, 10)}.json`;
    a.click(); URL.revokeObjectURL(url);
  }
}
