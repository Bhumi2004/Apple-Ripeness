import pandas as pd

# Load the DataFrame
df = pd.read_csv(r'C:/Users/BHUMI/Downloads/Sih.csv')

# Create styled DataFrame
styled_description = df.describe().T.style.background_gradient(axis=0, cmap='cubehelix')

# Save the styled DataFrame to an HTML file
html_output = styled_description.render()  # Render the styled DataFrame as HTML

# Write the HTML output to a file
with open('styled_description.html', 'w') as f:
    f.write(html_output)

print("Styled DataFrame saved as 'styled_description.html'")
