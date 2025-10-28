# Get It Started

Example of how to run the script for computing the churn and survival rates (*compute_rates.py*): 

```
# Before running the script, make sure to load all datasets according to the HuggingFace page.

python compute_rates.py --window 3w --limit 5000 --agent 0 --save your_location --start 3 --end 6
```
```window```: the time interval for which to calculate the code evolution
```limit```: the amount of commits for which to calculate the code evolution
```agent```: signal to filter commits according to each agent (0 = Claude, 1 = Codex, 2 = Copilot, 3 = Devin, 4 = Human, 5 = Jules) 
```save```: path where you want to save the results 
```start```: when multiple GitHub tokens are passed, this represents the token to start from 
```end```: when multiple GitHub tokens are passed, this represnts the token to end to 
