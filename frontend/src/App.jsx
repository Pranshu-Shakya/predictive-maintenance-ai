import { useState } from "react";

const API_URL = import.meta.env.VITE_BACKEND_API_URL;

const initialForm = {
	Temperature: 70,
	Vibration: 3.5,
	Pressure: 8,
	RPM: 1450,
	Operating_Hours: 4000,
	Flow_Rate: 115,
};

function App() {
	const [form, setForm] = useState(initialForm);
	const [result, setResult] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState("");

	const handleChange = (event) => {
		const { name, value } = event.target;

		setForm((previous) => ({
			...previous,
			[name]: value,
		}));
	};

	const analyzeMachine = async () => {
		setLoading(true);
		setError("");
		setResult(null);

		try {
			const payload = {
				Temperature: Number(form.Temperature),
				Vibration: Number(form.Vibration),
				Pressure: Number(form.Pressure),
				RPM: Number(form.RPM),
				Operating_Hours: Number(form.Operating_Hours),
				Flow_Rate: Number(form.Flow_Rate),
			};

			const response = await fetch(`${API_URL}/diagnose`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(payload),
			});

			if (!response.ok) {
				throw new Error("Failed to analyze machine");
			}

			const data = await response.json();

			setResult(data);
		} catch (err) {
			console.error(err);
			setError("Unable to connect to the backend. Make sure FastAPI is running.");
		} finally {
			setLoading(false);
		}
	};

	const probability = result ? Math.round(result.failure_probability * 100) : 0;

	return (
		<div className="min-h-screen bg-slate-50">
			{/* Header */}
			<header className="border-b bg-white">
				<div className="mx-auto max-w-7xl px-6 py-5">
					<div className="flex items-center justify-between">
						<div>
							<h1 className="text-2xl font-bold text-slate-900">
								AI Predictive Maintenance
							</h1>

							<p className="mt-1 text-sm text-slate-500">
								ML-powered machine health monitoring and AI troubleshooting
							</p>
						</div>

						<div className="rounded-full bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-700">
							System Online
						</div>
					</div>
				</div>
			</header>

			{/* Main */}
			<main className="mx-auto max-w-7xl px-6 py-8">
				<div className="grid gap-6 lg:grid-cols-3">
					{/* Sensor Input */}
					<section className="rounded-2xl border bg-white p-6 shadow-sm lg:col-span-1">
						<h2 className="text-lg font-semibold text-slate-900">
							Machine Parameters
						</h2>

						<p className="mt-1 text-sm text-slate-500">
							Enter current machine sensor readings.
						</p>

						<div className="mt-6 space-y-4">
							<InputField
								label="Temperature"
								name="Temperature"
								value={form.Temperature}
								unit="°C"
								onChange={handleChange}
							/>

							<InputField
								label="Vibration"
								name="Vibration"
								value={form.Vibration}
								unit="mm/s"
								onChange={handleChange}
							/>

							<InputField
								label="Pressure"
								name="Pressure"
								value={form.Pressure}
								unit="bar"
								onChange={handleChange}
							/>

							<InputField
								label="RPM"
								name="RPM"
								value={form.RPM}
								unit="rpm"
								onChange={handleChange}
							/>

							<InputField
								label="Operating Hours"
								name="Operating_Hours"
								value={form.Operating_Hours}
								unit="hours"
								onChange={handleChange}
							/>

							<InputField
								label="Flow Rate"
								name="Flow_Rate"
								value={form.Flow_Rate}
								unit="L/min"
								onChange={handleChange}
							/>
						</div>

						<button
							onClick={analyzeMachine}
							disabled={loading}
							className="mt-6 w-full rounded-xl bg-slate-900 px-4 py-3 font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
						>
							{loading ? "Analyzing..." : "Analyze Machine"}
						</button>

						{error && (
							<div className="mt-4 rounded-xl bg-red-50 p-4 text-sm text-red-700">
								{error}
							</div>
						)}
					</section>

					{/* Results */}
					<section className="space-y-6 lg:col-span-2">
						{/* Empty state */}
						{!result && !loading && (
							<div className="flex min-h-[400px] items-center justify-center rounded-2xl border bg-white p-8 text-center shadow-sm">
								<div>
									<div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-slate-100 text-2xl">
										AI
									</div>

									<h2 className="mt-5 text-xl font-semibold text-slate-900">
										Ready for Analysis
									</h2>

									<p className="mx-auto mt-2 max-w-md text-sm text-slate-500">
										Enter machine parameters and run an analysis to receive an
										ML-based risk prediction and AI troubleshooting report.
									</p>
								</div>
							</div>
						)}

						{/* Loading */}
						{loading && (
							<div className="flex min-h-[400px] items-center justify-center rounded-2xl border bg-white shadow-sm">
								<div className="text-center">
									<div className="mx-auto h-10 w-10 animate-spin rounded-full border-4 border-slate-200 border-t-slate-800" />

									<p className="mt-4 font-medium text-slate-700">
										Analyzing machine...
									</p>

									<p className="mt-1 text-sm text-slate-500">
										Running ML prediction and retrieving maintenance knowledge
									</p>
								</div>
							</div>
						)}

						{/* Results */}
						{result && !loading && (
							<>
								{/* Summary cards */}
								<div className="grid gap-4 md:grid-cols-3">
									<MetricCard
										title="Failure Probability"
										value={`${probability}%`}
									/>

									<MetricCard
										title="Predicted Fault"
										value={result.predicted_fault}
									/>

									<MetricCard
										title="Machine Status"
										value={formatStatus(result.machine_status)}
									/>
								</div>

								{/* Probability */}
								<div className="rounded-2xl border bg-white p-6 shadow-sm">
									<div className="flex items-center justify-between">
										<div>
											<h2 className="font-semibold text-slate-900">
												Failure Risk
											</h2>

											<p className="text-sm text-slate-500">
												Probability of machine failure
											</p>
										</div>

										<span className="text-2xl font-bold text-slate-900">
											{probability}%
										</span>
									</div>

									<div className="mt-5 h-3 overflow-hidden rounded-full bg-slate-100">
										<div
											className="h-full rounded-full bg-slate-900 transition-all duration-700"
											style={{
												width: `${probability}%`,
											}}
										/>
									</div>
								</div>

								{/* Fault Classification */}
								<FaultProbabilityCard probabilities={result.fault_probabilities} />

								{/* AI Diagnosis */}
								<div className="rounded-2xl border bg-white p-6 shadow-sm">
									<div>
										<h2 className="text-lg font-semibold text-slate-900">
											AI Troubleshooting Report
										</h2>

										<p className="mt-1 text-sm text-slate-500">
											Generated using ML prediction and retrieved maintenance
											knowledge
										</p>
									</div>

									{/* Summary */}
									<div className="mt-6 rounded-xl bg-slate-50 p-5">
										<p className="text-sm leading-7 text-slate-700">
											{result.ai_diagnosis.summary}
										</p>
									</div>

									{/* Why Prediction */}
									<ReportSection
										title="Why This Prediction Was Made"
										items={result.ai_diagnosis.why_prediction}
									/>

									{/* Possible Causes */}
									<ReportSection
										title="Possible Causes"
										items={result.ai_diagnosis.possible_causes}
									/>

									{/* Inspection Steps */}
									<ReportSection
										title="Recommended Inspection Steps"
										items={result.ai_diagnosis.inspection_steps}
									/>

									{/* Corrective Actions */}
									<ReportSection
										title="Corrective Actions"
										items={result.ai_diagnosis.corrective_actions}
									/>

									{/* Safety Note */}
									<div className="mt-6 rounded-xl bg-amber-50 p-4">
										<p className="text-sm font-semibold text-amber-800">
											Safety Note
										</p>

										<p className="mt-1 text-sm leading-6 text-amber-700">
											{result.ai_diagnosis.safety_note}
										</p>
									</div>
								</div>

								{/* Sources */}
								{result.sources?.length > 0 && (
									<div className="rounded-2xl border bg-white p-6 shadow-sm">
										<h2 className="font-semibold text-slate-900">
											Knowledge Sources
										</h2>

										<div className="mt-4 flex flex-wrap gap-2">
											{result.sources.map((source) => (
												<span
													key={source}
													className="rounded-full bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-600"
												>
													{source}
												</span>
											))}
										</div>
									</div>
								)}
							</>
						)}
					</section>
				</div>
			</main>
		</div>
	);
}

function InputField({ label, name, value, unit, onChange }) {
	return (
		<div>
			<label className="mb-1.5 block text-sm font-medium text-slate-700">{label}</label>

			<div className="relative">
				<input
					type="number"
					step="any"
					name={name}
					value={value}
					onChange={onChange}
					className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 pr-16 text-slate-900 outline-none transition focus:border-slate-500 focus:ring-2 focus:ring-slate-100"
				/>

				<span className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-slate-400">
					{unit}
				</span>
			</div>
		</div>
	);
}

function MetricCard({ title, value }) {
	return (
		<div className="rounded-2xl border bg-white p-5 shadow-sm">
			<p className="text-sm text-slate-500">{title}</p>

			<p className="mt-2 text-xl font-bold text-slate-900">{value}</p>
		</div>
	);
}

function ReportSection({ title, items = [] }) {
	return (
		<div className="mt-6">
			<h3 className="font-semibold text-slate-900">{title}</h3>

			<ul className="mt-3 space-y-2">
				{items.map((item, index) => (
					<li key={index} className="flex gap-3 text-sm leading-6 text-slate-600">
						<span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-400" />

						<span>{item}</span>
					</li>
				))}
			</ul>
		</div>
	);
}

function FaultProbabilityCard({ probabilities = {} }) {
	return (
		<div className="rounded-2xl border bg-white p-6 shadow-sm">
			<div>
				<h2 className="text-lg font-semibold text-slate-900">Fault Classification</h2>

				<p className="mt-1 text-sm text-slate-500">Multiclass ML model confidence</p>
			</div>

			<div className="mt-6 space-y-5">
				{Object.entries(probabilities)
					.sort(([, a], [, b]) => b - a)
					.map(([fault, probability]) => {
						const percentage = Math.round(probability * 100);

						return (
							<div key={fault}>
								<div className="mb-2 flex items-center justify-between">
									<span className="text-sm font-medium text-slate-700">
										{fault}
									</span>

									<span className="text-sm font-semibold text-slate-900">
										{percentage}%
									</span>
								</div>

								<div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
									<div
										className="h-full rounded-full bg-slate-900 transition-all duration-700"
										style={{
											width: `${percentage}%`,
										}}
									/>
								</div>
							</div>
						);
					})}
			</div>
		</div>
	);
}

function formatStatus(status) {
	return status
		.toLowerCase()
		.replace("_", " ")
		.replace(/\b\w/g, (char) => char.toUpperCase());
}

export default App;
