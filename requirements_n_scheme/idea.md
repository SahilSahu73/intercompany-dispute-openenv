# Architectural Blueprint for Multi-Agent Financial Dispute Resolution

## 1. Project Overview and Aim

**The Idea:** This project builds an autonomous, multi-agent artificial intelligence system that acts as an expert corporate financial controller. It is designed to navigate and resolve complex intercompany accounting disputes—such as mismatched invoices, currency exchange discrepancies, and supply chain liability conflicts—across a simulated multinational enterprise environment.

**Our Aim:** The primary goal is to achieve a mathematically perfect "zero-balance" elimination state across a multi-entity ledger. We aim to deploy a centralized "Orchestrator" agent that can autonomously perceive unbalanced accounts, consult specialized sub-agents for legal and tax context, and execute mathematically sound journal entries to balance the books without human intervention. 

## 2. Learning Trajectory and Defined Tasks

The environment is structured as a curriculum, scaling in complexity to train the agent progressively.

### Task 1: Easy (Baseline Batch Matching)
*   **Description:** The environment presents 1,000 structured intercompany transactions between a US Parent and a UK Subsidiary. The data is clean, and the majority are perfect 1-to-1 matches.
*   **Output Expectation:** The agent must accurately link corresponding debit and credit transaction IDs and execute the final elimination.
*   **Special Requirements:** The agent must demonstrate high-throughput procedural execution, repeatedly calling database functions to clear the backlog efficiently without hallucinating parameters or timing out.

### Task 2: Medium (Contextual Ambiguity & Currency Variance)
*   **Description:** Transactions lack explicit intercompany IDs and feature noisy text. A 30-day payment delay introduces a foreign exchange (FX) variance between GBP and USD. 
*   **Output Expectation:** The agent must extract vendor details from unstructured text, identify the exact FX discrepancy, post a structured adjustment for the FX loss/gain, and then successfully match the pair.
*   **Special Requirements:** The agent is not allowed to guess exchange rates; it must actively query a simulated Treasury API (via MCP) to calculate the deterministic mathematical variance before posting any adjustments.

### Task 3: Hard (Adversarial Dispute Resolution)
*   **Description:** A multi-hop supply chain scenario where goods are damaged in transit between two subsidiaries. The ledgers show massive, cascading imbalances because the receiving subsidiary refuses to recognize the payable.
*   **Output Expectation:** The agent must retrieve the transit contract, determine legal liability based on shipping terms, force the liable entity's ledger to recognize the payable and inventory loss, and process the final elimination.
*   **Special Requirements:** The primary agent must consult a Legal Sub-Agent to interpret the International Commercial Terms (e.g., CIF vs. FOB) to establish liability before taking any ledger action.

## 3. Agent Definitions and Task Distribution

To maintain stability and prevent race conditions, the system utilizes a strict maximum of three agents.

1.  **The Primary Agent (Enterprise Consolidation Orchestrator):** The only agent with write-access to the environment. It queries ledgers, identifies discrepancies, delegates specialized analysis to sub-agents, and executes the final financial adjustments and eliminations.
2.  **Sub-Agent 1 (Legal & Compliance Analyst):** A read-only expert agent. Its sole task is to ingest unstructured contracts, parse legal jargon (e.g., Incoterms, transit liability), and return a structured JSON response declaring which corporate entity is financially liable for a specific dispute.
3.  **Sub-Agent 2 (Tax & Treasury Specialist):** A read-only expert agent. It handles complex financial queries, retrieves historical foreign exchange rates, and performs transfer pricing margin calculations to ensure intercompany true-ups align with regulatory standards.

## 4. Architecture and Step-by-Step Flow

The system employs a **Client-Server Hub-and-Spoke Model** powered by the Model Context Protocol (MCP). 
*   **The Hub:** The Primary Agent acts as the MCP Client. 
*   **The Spokes:** The Sub-Agents, the ERP Database, and the Treasury APIs are wrapped as independent MCP Servers.

### Step-by-Step Task Iteration:
1.  **Perceive (Hub):** The Primary Agent queries the ERP MCP Server and discovers a $50,000 mismatch between Entity A and Entity B due to damaged freight.
2.  **Delegate (Spoke):** The Primary Agent invokes the `ask_legal_analyst` MCP tool, passing the unstructured transit contract text as the payload.
3.  **Analyze (Spoke):** The Legal Sub-Agent processes the contract, determines it was a "CIF" agreement, and concludes Entity B is liable.
4.  **Synthesize (Hub):** The Primary Agent receives the JSON response from the Legal Sub-Agent and formulates an accounting resolution.
5.  **Execute (Environment):** The Primary Agent uses the `Post_Adjustment` action to force Entity B to recognize the expense, then uses `Execute_Elimination` to clear the matched balance.

## 5. Action Spaces, States, and MCP Implementations

### State Space ($S$)
The state is the current environmental context passed to the agent at step $t$:
*   `unmatched_transactions`: List of dictionaries containing open ledger entries.
*   `context_documents`: Retrieved contracts, emails, or invoices.
*   `agent_memory`: The log of previous tool calls and sub-agent responses.

### Environment Action Space ($A$)
These actions directly alter the environment and progress the episode:
*   `Execute_Match`
    *   *Inputs:* `debit_txn_id` (str), `credit_txn_id` (str)
    *   *Output:* Boolean success/fail. Only succeeds if absolute values are identical.
*   `Post_Adjustment`
    *   *Inputs:* `entity_id` (str), `debit_acct` (str), `credit_acct` (str), `amount` (float)
    *   *Output:* Updated ledger state. Fails if debit!= credit.
*   `Execute_Elimination`
    *   *Inputs:* `parent_entity_id` (str), `matched_pair_id` (str)
    *   *Output:* Removes the matched pair from the active board.

### MCP Tool Implementations (Information Gathering)
*   `query_ledger`: Inputs: `entity_id`, `account_code`. Output: Current numerical balance.
*   `fetch_document`: Inputs: `transaction_id`. Output: Unstructured text of the associated invoice/contract.
*   `ask_legal_agent`: Inputs: `document_text`, `query`. Output: Structured JSON liability determination.
*   `calculate_fx`: Inputs: `source_currency`, `target_currency`, `date`, `amount`. Output: Float value of the converted currency.

## 6. Reward Function Formulation

The environment uses a Process-Supervised Reward Model (PRM) to grade both the agent's final outcome and its step-by-step reasoning trajectory. This ensures the agent learns *how* to solve the problem, rather than randomly guessing actions to force a balanced ledger.

### 6.1 Step-Level Reward (Procedural Correctness)
At each step $t$, the agent receives a dense reward based on the logical sequence of its actions. 

$$r_{step}(s_t, a_t) = w_1 \cdot C_{proc} - \delta$$

*   **Inputs for calculation:** Current action $a_t$, previous actions list, available evidence.
*   $C_{proc} \in \{-1, 0, 1\}$: Procedural validation. If the agent calls `Post_Adjustment` without first using `fetch_document` or an MCP sub-agent to gather evidence, $C_{proc} = -1$. If it follows a logical sequence (e.g., fetch -> ask sub-agent -> adjust), $C_{proc} = 1$.
*   $\delta$: A small constant step penalty (e.g., 0.01) to penalize inefficiency and looping.

### 6.2 Trajectory-Level Reward (Final State Evaluation)
At the end of the episode, an independent "Audit Grader" script evaluates the final database state.

$$r_{traj} = B \cdot \mathbb{I}\left(\sum D = \sum C\right) + \sum_{i=1}^{N} \omega_i \cdot v_i - P \cdot E_{err}$$

**Inputs for calculation:** Final ledger database state, ground truth checklist array.

*   **Terminal Balance Bonus ($B \cdot \mathbb{I}$):** $\mathbb{I}$ is an indicator function that equals 1 if the fundamental accounting equation ($\sum \text{Debits} = \sum \text{Credits}$) is perfectly balanced across the consolidated ledger, and 0 otherwise. $B$ is a massive terminal bonus (e.g., +100).
*   **Checklist Vector ($\sum \omega_i \cdot v_i$):** A deterministic array of boolean checks $v_i \in \{0, 1\}$. 
    *   *Check 1:* Did the agent use the correct historical FX rate? (Validated against a hidden ground-truth variable).
    *   *Check 2:* Was the adjustment posted to the correct entity based on the Incoterms? (Validated against the scenario ground-truth).
    *   $\omega_i$ represents the point weight of each specific check.
*   **Catastrophic Penalty ($P \cdot E_{err}$):** $E_{err}$ counts the number of unauthorized actions (e.g., hallucinating an account code that doesn't exist, or posting an unbalanced journal entry). $P$ is a heavy multiplier (e.g., -50) that severely punishes specification gaming.

By combining the dense step-level feedback with the strict mathematical boundaries of the trajectory reward, the reinforcement learning algorithm will securely train the agent to act as a meticulous, efficient, and legally compliant financial controller.
