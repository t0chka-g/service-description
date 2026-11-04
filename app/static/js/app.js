const GENERATE_PATH = "/api/v1/products/generate";
const TITLE_LIMIT = 70;
const DESCRIPTION_MIN = 800;
const DESCRIPTION_MAX = 1200;

const DEMO = {
  name: "Манометр демонстрационный МП-Демо-01",
  manufacturer: "Учебный стенд «Сигнал»",
  description:
    "Учебный пример, не паспорт реального прибора. " +
    "Диапазон показаний 0...1,6 МПа. Класс точности 1,5. " +
    "Присоединение M20x1,5. Степень защиты IP54.",
  specs: [
    { name: "Диапазон показаний", value: "0...1,6 МПа" },
    { name: "Класс точности", value: "1,5" },
    { name: "Присоединение", value: "M20x1,5" },
    { name: "Степень защиты", value: "IP54" },
  ],
};

const WARNING_LABELS = {
  conflict: "Конфликт между характеристиками и описанием",
  missing_field: "Нет подтвержденного значения",
  insufficient_data: "Данных недостаточно для полной карточки",
  needs_review: "Часть карточки не прошла проверку",
  generation_failed: "Модель не вернула разбираемый ответ",
  description_too_short: "Описание короче нормы",
  description_too_long: "Описание длиннее нормы",
  description_rejected: "Описание не прошло проверку",
  search_queries_insufficient: "Недостаточно поисковых запросов",
};

const NO_DESCRIPTION = "Описание не сформировано. Причина указана в предупреждениях ниже.";
const NO_COMMERCIAL = "Коммерческий текст не сформирован.";

const SOURCE_LABELS = {
  specs: "Из характеристик",
  description: "Из описания",
};

const STATUS_LABELS = {
  complete: "Полная карточка",
  insufficient_data: "Недостаточно данных",
  needs_review: "Нужна проверка",
};

const FIELD_LABELS = {
  name: "Название прибора",
  manufacturer: "Производитель",
  description: "Техническое описание",
  specs: "Характеристики",
};

const form = document.querySelector("#product-form");
const result = document.querySelector("#result");
const specList = document.querySelector("#spec-list");
const generateButton = document.querySelector("#generate-button");
const nameInput = document.querySelector("#product-name");
const manufacturerInput = document.querySelector("#product-manufacturer");
const descriptionInput = document.querySelector("#product-description");
const formAlert = document.querySelector("#form-alert");
const copyStatus = document.querySelector("#copy-status");

const state = {
  specs: [{ name: "", value: "" }],
  rowErrors: [""],
  inFlight: false,
};

let requestToken = 0;
let abortController = null;
let pendingHighlight = -1;

if (
  form instanceof HTMLFormElement &&
  result instanceof HTMLElement &&
  specList instanceof HTMLElement &&
  generateButton instanceof HTMLButtonElement &&
  nameInput instanceof HTMLInputElement &&
  manufacturerInput instanceof HTMLInputElement &&
  descriptionInput instanceof HTMLTextAreaElement &&
  formAlert instanceof HTMLElement
) {
  bindForm();
  renderSpecs();
  updateDescriptionCount();
  renderEmpty();
}

function bindForm() {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void submit();
  });
  document.querySelector("#add-spec")?.addEventListener("click", addSpec);
  document.querySelector("#fill-demo")?.addEventListener("click", fillDemo);
  document.querySelector("#clear-form")?.addEventListener("click", clearForm);
  specList.addEventListener("input", onSpecInput);
  specList.addEventListener("click", onSpecClick);
  for (const input of [nameInput, manufacturerInput, descriptionInput]) {
    input.addEventListener("input", () => clearControl(input));
  }
  descriptionInput.addEventListener("input", updateDescriptionCount);
}

function addSpec() {
  state.specs.push({ name: "", value: "" });
  state.rowErrors.push("");
  pendingHighlight = state.specs.length - 1;
  renderSpecs();
  pendingHighlight = -1;
  const inputs = specList.querySelectorAll("input[data-field='name']");
  const last = inputs[inputs.length - 1];
  if (last instanceof HTMLInputElement) last.focus();
}

function onSpecInput(event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement)) return;
  const row = target.closest("[data-index]");
  if (!(row instanceof HTMLElement)) return;
  const index = Number(row.dataset.index);
  const field = target.dataset.field;
  if (!state.specs[index] || (field !== "name" && field !== "value")) return;
  state.specs[index][field] = target.value;
  clearControl(target);
  row.classList.remove("is-invalid");
  const message = row.querySelector(".field-error");
  if (message instanceof HTMLElement) {
    message.textContent = "";
    message.hidden = true;
  }
}

function onSpecClick(event) {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const button = target.closest("[data-remove]");
  if (!(button instanceof HTMLButtonElement)) return;
  const index = Number(button.dataset.remove);
  state.specs.splice(index, 1);
  state.rowErrors.splice(index, 1);
  if (!state.specs.length) {
    state.specs.push({ name: "", value: "" });
    state.rowErrors.push("");
  }
  renderSpecs();
  const focusAt = Math.min(index, state.specs.length - 1);
  const row = specList.querySelector(`[data-index="${focusAt}"] input[data-field="name"]`);
  if (row instanceof HTMLInputElement) row.focus();
}

function fillDemo() {
  nameInput.value = DEMO.name;
  manufacturerInput.value = DEMO.manufacturer;
  descriptionInput.value = DEMO.description;
  state.specs = DEMO.specs.map((item) => ({ ...item }));
  state.rowErrors = state.specs.map(() => "");
  clearFormErrors();
  renderSpecs();
  updateDescriptionCount();
}

function clearForm() {
  requestToken += 1;
  abortController?.abort();
  state.inFlight = false;
  updateGenerateButton();
  nameInput.value = "";
  manufacturerInput.value = "";
  descriptionInput.value = "";
  state.specs = [{ name: "", value: "" }];
  state.rowErrors = [""];
  clearFormErrors();
  renderSpecs();
  updateDescriptionCount();
  renderEmpty();
  nameInput.focus();
}

function clearFormErrors() {
  formAlert.hidden = true;
  formAlert.textContent = "";
  for (const input of [nameInput, manufacturerInput, descriptionInput]) {
    clearControl(input);
  }
  const specsError = document.querySelector("#specs-error");
  if (specsError instanceof HTMLElement) {
    specsError.hidden = true;
    specsError.textContent = "";
  }
}

function clearControl(input) {
  input.removeAttribute("aria-invalid");
  input.removeAttribute("aria-describedby");
  const error = document.getElementById(`${input.id}-error`);
  if (error instanceof HTMLElement) {
    error.hidden = true;
    error.textContent = "";
  }
}

function updateDescriptionCount() {
  const node = document.querySelector("#description-count");
  if (!(node instanceof HTMLElement)) return;
  const length = descriptionInput.value.length;
  node.textContent = `необязательно · ${length} ${plural(length, "знак", "знака", "знаков")}`;
}

function renderSpecs() {
  specList.replaceChildren();
  state.specs.forEach((spec, index) => {
    specList.append(buildSpecRow(spec, index));
  });
  const count = document.querySelector("#spec-count");
  if (count instanceof HTMLElement) {
    const total = state.specs.length;
    count.textContent = `${total} ${plural(total, "строка", "строки", "строк")}`;
  }
}

function buildSpecRow(spec, index) {
  const row = document.createElement("div");
  row.className = "editor-row";
  if (index === pendingHighlight) row.classList.add("is-new");
  row.dataset.index = String(index);

  const nameField = editorField(
    "name",
    spec.name,
    "Параметр",
    `Название параметра ${index + 1}`,
    "Диапазон измерений",
  );
  const valueField = editorField(
    "value",
    spec.value,
    "Значение",
    `Значение параметра ${index + 1}`,
    "0–1,6 МПа",
  );
  row.append(nameField, valueField, editorRemove(index));

  const message = state.rowErrors[index];
  const nameInputNode = nameField.querySelector("input");
  if (message && nameInputNode instanceof HTMLInputElement) {
    row.classList.add("is-invalid");
    nameInputNode.setAttribute("aria-invalid", "true");
    const error = document.createElement("p");
    error.className = "field-error";
    error.id = `spec-error-${index}`;
    error.textContent = message;
    nameInputNode.setAttribute("aria-describedby", error.id);
    row.append(error);
  }
  return row;
}

function editorField(field, value, label, ariaLabel, placeholder) {
  const wrap = document.createElement("label");
  wrap.className = "editor-cell";
  const caption = document.createElement("span");
  caption.className = "cell-label";
  caption.textContent = label;
  caption.setAttribute("aria-hidden", "true");
  const input = document.createElement("input");
  input.type = "text";
  input.dataset.field = field;
  input.value = value;
  input.placeholder = placeholder;
  input.spellcheck = false;
  input.autocomplete = "off";
  input.setAttribute("aria-label", ariaLabel);
  wrap.append(caption, input);
  return wrap;
}

function editorRemove(index) {
  const wrap = document.createElement("div");
  wrap.className = "editor-action";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "button button-quiet";
  button.dataset.remove = String(index);
  button.textContent = "Удалить";
  button.setAttribute("aria-label", `Удалить характеристику ${index + 1}`);
  wrap.append(button);
  return wrap;
}

async function submit() {
  if (state.inFlight || !(generateButton instanceof HTMLButtonElement)) return;
  const payload = readPayload();
  if (!payload) return;

  const token = ++requestToken;
  abortController?.abort();
  abortController = new AbortController();
  state.inFlight = true;
  updateGenerateButton();
  renderLoading();

  try {
    const response = await fetch(GENERATE_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: abortController.signal,
    });
    if (token !== requestToken) return;
    if (response.status === 422) {
      const body = await response.json();
      showValidation(body?.detail);
      renderError("Проверьте выделенные поля и повторите запрос.");
      return;
    }
    if (!response.ok) {
      renderError(await errorMessage(response));
      return;
    }
    renderCard(await response.json());
  } catch (error) {
    if (token !== requestToken || isAbort(error)) return;
    renderError("Нет связи с сервером. Повторите запрос.");
  } finally {
    if (token === requestToken) {
      state.inFlight = false;
      updateGenerateButton();
    }
  }
}

function readPayload() {
  clearFormErrors();
  state.rowErrors = state.specs.map(() => "");
  const name = nameInput.value.trim();
  const messages = [];
  if (!name) {
    messages.push("Укажите название прибора.");
    setControlError(nameInput, "Укажите название прибора.");
  }

  const specs = {};
  const seen = new Set();
  state.specs.forEach((spec, index) => {
    const specName = spec.name.trim();
    const specValue = spec.value;
    if (!specName && !specValue.trim()) return;
    if (!specName) {
      const message = "Укажите название характеристики.";
      state.rowErrors[index] = message;
      messages.push(message);
      return;
    }
    const key = specName.toLocaleLowerCase("ru");
    if (seen.has(key)) {
      const message = "Такая характеристика уже есть.";
      state.rowErrors[index] = message;
      messages.push(message);
      return;
    }
    seen.add(key);
    specs[specName] = specValue;
  });
  renderSpecs();

  if (messages.length) {
    formAlert.textContent = messages.filter((item, index) => messages.indexOf(item) === index).join(" ");
    formAlert.hidden = false;
    const invalid = form.querySelector("[aria-invalid='true']");
    if (invalid instanceof HTMLElement) invalid.focus();
    return null;
  }

  return {
    name,
    manufacturer: manufacturerInput.value.trim(),
    description: descriptionInput.value.trim(),
    specs,
  };
}

function showValidation(detail) {
  const messages = [];
  const items = Array.isArray(detail) ? detail : [];
  for (const item of items) {
    const loc = Array.isArray(item.loc) ? item.loc.map(String) : [];
    const field = loc.find((part) => part in FIELD_LABELS);
    const label = field ? FIELD_LABELS[field] : "Данные";
    messages.push(`${label}: проверьте значение.`);
    if (field === "name") setControlError(nameInput, "Проверьте название.");
    if (field === "manufacturer") setControlError(manufacturerInput, "Проверьте производителя.");
    if (field === "description") setControlError(descriptionInput, "Проверьте описание.");
    if (field === "specs") {
      const specsError = document.querySelector("#specs-error");
      if (specsError instanceof HTMLElement) {
        specsError.hidden = false;
        specsError.textContent = "Проверьте список характеристик.";
      }
    }
  }
  formAlert.textContent = messages.length
    ? messages.filter((item, index) => messages.indexOf(item) === index).join(" ")
    : "Проверьте данные формы.";
  formAlert.hidden = false;
}

function setControlError(input, message) {
  input.setAttribute("aria-invalid", "true");
  input.setAttribute("aria-describedby", `${input.id}-error`);
  const error = document.getElementById(`${input.id}-error`);
  if (error instanceof HTMLElement) {
    error.hidden = false;
    error.textContent = message;
  }
}

function updateGenerateButton() {
  generateButton.disabled = state.inFlight;
  generateButton.setAttribute("aria-busy", state.inFlight ? "true" : "false");
  generateButton.textContent = state.inFlight ? "Генерируем…" : "Сгенерировать карточку";
}

function renderEmpty() {
  setResultState("empty");
  const box = document.createElement("div");
  box.className = "empty-state";
  box.append(
    element("p", "Карточка появится здесь", "lead"),
    element(
      "p",
      "Заполните исходные данные и запустите сборку. До этого текст не создается.",
      "note",
    ),
  );
  mount(box, false);
}

function renderLoading() {
  setResultState("loading");
  const box = document.createElement("div");
  box.className = "loading";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  box.append(
    spinner,
    element("h3", "Генерируем карточку товара", "lead"),
    element(
      "p",
      "Сверяем характеристики и собираем черновик. Это может занять некоторое время.",
      "note",
    ),
  );
  mount(box, true);
}

function renderError(message) {
  setResultState("error");
  const box = document.createElement("div");
  box.className = "error-state";
  const retry = document.createElement("button");
  retry.type = "button";
  retry.className = "button button-secondary";
  retry.textContent = "Повторить";
  retry.addEventListener("click", () => {
    void submit();
  });
  box.append(element("h3", "Запрос не выполнен", "lead"), element("p", message, "callout"), retry);
  mount(box, true);
}

function isPartial(card) {
  return card.status === "insufficient_data" || card.status === "needs_review";
}

function renderCard(card) {
  const partial = isPartial(card);
  setResultState(partial ? "insufficient_data" : "complete");
  const stack = document.createElement("div");
  stack.className = "result-stack";
  stack.append(verdict(card));
  stack.append(
    textBlock({
      title: "SEO-заголовок",
      text: card.seo_title,
      counter: titleCounter(card.seo_title),
      copyValue: () => card.seo_title,
      bodyClass: "seo-title",
    }),
  );
  stack.append(
    textBlock({
      title: "SEO-описание",
      text: card.seo_description,
      emptyText: NO_DESCRIPTION,
      counter: descriptionCounter(card),
      counterTone: partial ? "short" : "",
      copyValue: () => card.seo_description,
      bodyClass: "prose",
    }),
  );
  stack.append(specificationBlock(card.specifications));
  stack.append(
    listBlock("Поисковые запросы", card.search_queries, "Запросы не сформированы.", "query-list"),
  );
  stack.append(
    textBlock({
      title: "Коммерческий текст",
      text: card.commercial_block,
      emptyText: NO_COMMERCIAL,
      copyValue: () => card.commercial_block,
      bodyClass: "prose",
    }),
  );
  stack.append(
    listBlock(
      "Недостающие данные",
      card.missing_fields,
      "Неподтвержденных характеристик нет.",
      "missing-list",
      "Эти параметры не подтверждены и в текст не вошли.",
    ),
  );
  stack.append(warningBlock(card.warnings));
  stack.append(...exportNodes(card));
  mount(stack, true);
}

function verdict(card) {
  const partial = isPartial(card);
  const review = card.status === "needs_review";
  const header = document.createElement("div");
  header.className = partial ? "verdict verdict-partial" : "verdict";
  header.append(
    element(
      "p",
      review ? "Нужна проверка" : partial ? "Недостаточно данных" : "Карточка собрана",
      "verdict-label",
    ),
    element(
      "p",
      review
        ? "Часть полей не прошла проверку и не опубликована. Подтвержденные характеристики сохранены."
        : partial
          ? "Полное описание не собрано. Ниже только подтвержденные сведения — это не ошибка запроса."
          : "Полная карточка. Проверьте формулировки перед публикацией.",
      "verdict-text",
    ),
  );
  return header;
}

function textBlock({ title, text, emptyText, counter, counterTone, copyValue, bodyClass }) {
  const section = document.createElement("section");
  section.className = "result-section";
  const head = document.createElement("div");
  head.className = "section-head";
  const titles = document.createElement("div");
  titles.className = "caption-block";
  titles.append(element("h3", title));
  if (counter) {
    const tone = counterTone ? `counter ${counterTone}` : "counter";
    titles.append(element("p", counter, tone));
  }
  head.append(titles);
  const hasText = typeof text === "string" && text.trim() !== "";
  if (copyValue && hasText) head.append(copyButton("Копировать", copyValue));
  const value = hasText ? text : emptyText || "—";
  section.append(head, element("p", value, hasText ? bodyClass || "prose" : "note"));
  return section;
}

function specificationBlock(items) {
  const section = document.createElement("section");
  section.className = "result-section";
  const heading = element("h3", "Технические характеристики");
  heading.id = "confirmed-specs";
  section.append(heading);
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) {
    section.append(element("p", "Нет подтвержденных характеристик.", "note"));
    return section;
  }
  const table = document.createElement("table");
  table.className = "spec-table";
  table.setAttribute("aria-labelledby", heading.id);
  const head = document.createElement("tr");
  for (const label of ["Параметр", "Значение", "Источник"]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.append(cell);
  }
  const thead = document.createElement("thead");
  thead.append(head);
  const body = document.createElement("tbody");
  for (const item of rows) {
    const row = document.createElement("tr");
    const cells = [
      item.name,
      item.value,
      SOURCE_LABELS[item.source] || item.source || "",
    ];
    cells.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value ?? "";
      cell.dataset.label = ["Параметр", "Значение", "Источник"][index];
      if (index === 1) cell.className = "num";
      if (index === 2) cell.className = "source";
      row.append(cell);
    });
    body.append(row);
  }
  table.append(thead, body);
  section.append(table);
  return section;
}

function listBlock(title, values, emptyText, listClass, intro) {
  const section = document.createElement("section");
  section.className = "result-section";
  section.append(element("h3", title));
  const items = Array.isArray(values) ? values.filter((value) => String(value ?? "").trim()) : [];
  if (!items.length) {
    section.append(element("p", emptyText, "note"));
    return section;
  }
  if (intro) section.append(element("p", intro, "note"));
  const list = document.createElement(listClass === "query-list" ? "ol" : "ul");
  list.className = listClass;
  for (const value of items) list.append(element("li", String(value)));
  section.append(list);
  return section;
}

function warningBlock(warnings) {
  const section = document.createElement("section");
  section.className = "result-section";
  section.append(element("h3", "Предупреждения"));
  const items = Array.isArray(warnings) ? warnings : [];
  if (!items.length) {
    section.append(element("p", "Предупреждений нет.", "note"));
    return section;
  }
  const list = document.createElement("ul");
  list.className = "warning-list";
  for (const warning of items) list.append(warningItem(warning));
  section.append(list);
  return section;
}

function warningItem(warning) {
  const item = document.createElement("li");
  const code = element("p", WARNING_LABELS[warning.code] || warning.code || "Предупреждение", "warn-code");
  item.append(code);
  if (warning.field) item.append(element("p", warning.field, "warn-field"));
  if (warning.message) item.append(element("p", warning.message, "warn-message"));
  return item;
}

function exportNodes(card) {
  const bar = document.createElement("div");
  bar.className = "export-bar";
  const rawButton = document.createElement("button");
  rawButton.type = "button";
  rawButton.className = "button button-secondary";
  rawButton.textContent = "Показать JSON";
  rawButton.setAttribute("aria-expanded", "false");
  rawButton.setAttribute("aria-controls", "card-json");
  const raw = document.createElement("pre");
  raw.className = "raw-json";
  raw.id = "card-json";
  raw.hidden = true;
  raw.tabIndex = 0;
  raw.setAttribute("aria-label", "JSON карточки");
  raw.textContent = JSON.stringify(card, null, 2);
  rawButton.addEventListener("click", () => {
    raw.hidden = !raw.hidden;
    const open = !raw.hidden;
    rawButton.setAttribute("aria-expanded", open ? "true" : "false");
    rawButton.textContent = open ? "Скрыть JSON" : "Показать JSON";
    if (open) raw.focus();
  });
  bar.append(
    copyButton("Копировать Markdown", () => toMarkdown(card), false),
    copyButton("Копировать JSON", () => JSON.stringify(card, null, 2), false),
    rawButton,
  );
  return [raw, bar];
}

function copyButton(label, valueOf, quiet = true) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = quiet ? "button button-quiet copy-button" : "button button-secondary copy-button";
  button.textContent = label;
  button.addEventListener("click", () => {
    const value = typeof valueOf === "function" ? valueOf() : valueOf;
    void copyText(String(value ?? ""), button, label);
  });
  return button;
}

async function copyText(text, button, label) {
  try {
    await writeClipboard(text);
    button.textContent = "Скопировано";
    announce("Скопировано");
  } catch {
    button.textContent = "Не скопировано";
    announce("Не скопировано");
  }
  window.setTimeout(() => {
    if (button.isConnected) button.textContent = label;
  }, 1600);
}

function announce(message) {
  if (!(copyStatus instanceof HTMLElement)) return;
  copyStatus.textContent = "";
  copyStatus.textContent = message;
}

async function writeClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.left = "-9999px";
  document.body.append(area);
  area.select();
  const copied = document.execCommand("copy");
  area.remove();
  if (!copied) throw new Error("copy failed");
}

function toMarkdown(card) {
  const lines = [`# ${card.seo_title}`, "", card.seo_description || `_${NO_DESCRIPTION}_`, "", "## Технические характеристики", ""];
  if (card.specifications?.length) {
    lines.push("| Параметр | Значение | Источник |", "| --- | --- | --- |");
    for (const item of card.specifications) {
      const source = SOURCE_LABELS[item.source] || item.source || "";
      lines.push(`| ${mdCell(item.name)} | ${mdCell(item.value)} | ${mdCell(source)} |`);
    }
  } else {
    lines.push("Нет подтвержденных характеристик.");
  }
  lines.push("", "## Поисковые запросы", "");
  lines.push(...listLines(card.search_queries, "Запросы не сформированы."));
  lines.push("", "## Коммерческий текст", "", card.commercial_block || NO_COMMERCIAL, "", "## Недостающие данные", "");
  lines.push(...listLines(card.missing_fields, "Неподтвержденных характеристик нет."));
  lines.push("", "## Предупреждения", "");
  if (!card.warnings?.length) {
    lines.push("- Предупреждений нет.");
  } else {
    for (const warning of card.warnings) {
      const label = WARNING_LABELS[warning.code] || warning.code || "Предупреждение";
      const field = warning.field ? `: ${warning.field}` : "";
      const message = warning.message ? `. ${warning.message}` : "";
      lines.push(`- ${label}${field}${message}`);
    }
  }
  lines.push("", `Статус: ${STATUS_LABELS[card.status] || card.status}`);
  return lines.join("\n");
}

function listLines(values, emptyText) {
  if (!values?.length) return [`- ${emptyText}`];
  return values.map((value) => `- ${value}`);
}

function mdCell(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}

function titleCounter(title) {
  return `${String(title ?? "").length} из ${TITLE_LIMIT}`;
}

function descriptionCounter(card) {
  if (typeof card.seo_description !== "string") {
    return `нужно ${DESCRIPTION_MIN}–${DESCRIPTION_MAX} знаков`;
  }
  const length = card.seo_description.length;
  if (card.status === "insufficient_data" || card.status === "needs_review") {
    return `${length} ${plural(length, "знак", "знака", "знаков")} · для полной карточки нужно от ${DESCRIPTION_MIN}`;
  }
  return `${length} ${plural(length, "знак", "знака", "знаков")} · норма ${DESCRIPTION_MIN}–${DESCRIPTION_MAX}`;
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string" && body.detail.trim()) return body.detail;
  } catch {
    return "Не удалось сгенерировать карточку. Повторите запрос.";
  }
  return "Не удалось сгенерировать карточку. Повторите запрос.";
}

function isAbort(error) {
  return error instanceof DOMException && error.name === "AbortError";
}

function setResultState(stateName) {
  result.dataset.state = stateName;
  result.setAttribute("aria-busy", stateName === "loading" ? "true" : "false");
}

function mount(node, animate) {
  if (animate) node.classList.add("is-entering");
  result.replaceChildren(node);
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

function plural(count, one, few, many) {
  const n = Math.abs(count) % 100;
  const n1 = n % 10;
  if (n > 10 && n < 20) return many;
  if (n1 === 1) return one;
  if (n1 > 1 && n1 < 5) return few;
  return many;
}
