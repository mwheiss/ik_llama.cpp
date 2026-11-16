import { expect, test } from '@playwright/test';

test('home page loads correctly', async ({ page }) => {
	await page.route('**/props*', (route) =>
		route.fulfill({ json: { role: 'MODEL', model_path: 'test-model', modalities: {} } })
	);
	await page.route('**/v1/models', (route) =>
		route.fulfill({ json: { object: 'list', data: [{ id: 'test-model' }] } })
	);
	await page.route('**/tools', (route) => route.fulfill({ json: [] }));
	await page.goto('/index_llamacpp.html');
	// Wait for the greeting to become visible (stores need time to initialize)
	await expect(page.locator('h1', { hasText: /Hello there/ })).toBeVisible();
});
