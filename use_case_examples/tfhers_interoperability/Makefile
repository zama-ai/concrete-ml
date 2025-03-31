# Makefile tfhers_interoperability

# Useful for jupyter notebooks
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

TIME_NB="${USE_CASE_DIR}/time_notebook_execution.sh"

run_example: install_rust run_rust run_notebook

install_rust:
	@command -v cargo >/dev/null 2>&1 || ( \
		echo "🚀 Installing Rust..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		source $$HOME/.cargo/env; \
		echo "✅ Rust installed!" \
	)

run_rust:
	@bash -c 'source $$HOME/.cargo/env && cargo build --release'

run_notebook:
	@$(TIME_NB) AuthenticationWithDecisionTree.ipynb
