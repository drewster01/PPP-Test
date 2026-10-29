# Stores your SportsGameOdds key in your Windows user environment (SGO_KEY).
# The key is typed into a hidden prompt: it is never shown, logged, or written to a file.
# Run:  powershell -ExecutionPolicy Bypass -File set_key.ps1

$secure = Read-Host "Paste your SportsGameOdds API key (input hidden)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim()
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
if ([string]::IsNullOrWhiteSpace($key)) {
    Write-Host "No key entered; nothing changed."
    exit 1
}
[Environment]::SetEnvironmentVariable("SGO_KEY", $key, "User")
$env:SGO_KEY = $key
Write-Host ("Saved SGO_KEY for your user account ({0} characters). Restart the Claude app or open a new terminal so it is picked up." -f $key.Length)
