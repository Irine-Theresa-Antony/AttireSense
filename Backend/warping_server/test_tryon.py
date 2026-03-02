import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ---------- CONFIG ----------
FRONTEND_URL = "http://localhost:3000"
TEST_IMAGE_PATH = r"C:\Users\farzi\Desktop\fahmi\mainProject\AttireSense\Backend\warping_server\dataset\cloth\07367_00.jpg"
# ----------------------------

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.maximize_window()
driver.get(FRONTEND_URL)

wait = WebDriverWait(driver, 60)

# 1️⃣ Upload image
file_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='file']")))
file_input.send_keys(TEST_IMAGE_PATH)

time.sleep(2)

# 2️⃣ Open dropdown
dropdown_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Selected Model')]")))
dropdown_button.click()

time.sleep(1)

# 3️⃣ Select first model option
model_option = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'cursor-pointer')]")))
model_option.click()

time.sleep(1)

# 4️⃣ Click STYLE ME
style_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'STYLE ME')]")))
style_button.click()

# 5️⃣ Wait for result image
result_image = wait.until(EC.presence_of_element_located((By.XPATH, "//img[contains(@alt,'Result')]")))

print("✅ Try-on test passed. Result image loaded.")

time.sleep(5)
driver.quit()